// Copyright 2018, Intel Corporation

#include "tile/codegen/fc.h"
#include "base/util/any_factory_map.h"
#include "tile/codegen/tile.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT
using namespace math;    // NOLINT

// Check if
// 1) access 0..(n-last-1) are zero
// 2) access (n-last)..(n-1) are single index
bool ZeroAccesses(const Refinement& ref, size_t last) {
  size_t n_acc = ref.access.size();
  if (n_acc < last) {
    return false;
  }
  for (size_t i = 0; i < n_acc - last; ++i) {
    if (ref.access[i] != Affine()) {
      return false;
    }
  }
  for (size_t i = n_acc - last; i < n_acc; ++i) {
    const auto& acc_map = ref.access[i].getMap();
    if (acc_map.size() != 1) {
      return false;
    }
    if (acc_map.begin()->first == "" || acc_map.begin()->second != 1) {
      return false;
    }
  }
  return true;
}

// We process only the pattern A[...x] * B[...x, y], in which x * y >= threshold
void FullyConnected(const AliasMap& alias_map, Block* block, const proto::FullyConnectedPass& options) {
  std::string buf1d_name;
  std::string buf2d_name;
  std::string sbuf_name;
  std::string buf1d_idx;
  std::string sbuf_idx;
  std::string buf2d_first_idx;
  std::string buf2d_last_idx;
  for (const auto& ref : block->refs) {
    size_t n_acc = ref.access.size();
    if (IsReadDir(ref.dir)) {
      if (ZeroAccesses(ref, 1)) {
        const auto& acc_map = ref.access[n_acc - 1].getMap();
        buf1d_idx = acc_map.begin()->first;
        buf1d_name = ref.into();
      } else if (ZeroAccesses(ref, 2)) {
        const auto& acc_map0 = ref.access[n_acc - 2].getMap();
        buf2d_first_idx = acc_map0.begin()->first;
        const auto& acc_map1 = ref.access[n_acc - 1].getMap();
        buf2d_last_idx = acc_map1.begin()->first;
        buf2d_name = ref.into();
      }
    } else if (IsWriteDir(ref.dir)) {
      if (ZeroAccesses(ref, 1)) {
        const auto& acc_map = ref.access[n_acc - 1].getMap();
        sbuf_idx = acc_map.begin()->first;
        sbuf_name = ref.into();
      }
    }
  }
  if (buf1d_idx != buf2d_first_idx || sbuf_idx != buf2d_last_idx || buf1d_idx.empty() || sbuf_idx.empty()) {
    return;
  }
  IVLOG(3, "buf1d: " << buf1d_name << ", buf2d: " << buf2d_name << ", common index: " << buf1d_idx);
  if (block->exterior_shape(buf2d_name).sizes_product() < options.threshold()) {
    return;
  }
  // Determine the largest subgroup size.
  // The ranges of the index must be divisible by it.
  size_t subgroup_size = 0;
  auto idx0 = block->idx_by_name(buf1d_idx);
  auto idx1 = block->idx_by_name(buf2d_last_idx);
  for (int i = options.subgroup_sizes().size() - 1; i >= 0; --i) {
    size_t current_size = options.subgroup_sizes()[i];
    if ((current_size > subgroup_size) && (idx0->range % current_size == 0) && (idx1->range % current_size == 0)) {
      subgroup_size = current_size;
    }
  }
  if (subgroup_size == 0) {
    return;
  }
  IVLOG(3, "Subgroup size = " << subgroup_size);

  // All checks finished. Start transformation.
  TileShape tile(block->idxs.size());
  for (size_t i = 0; i < block->idxs.size(); ++i) {
    const auto& idx = block->idxs[i];
    tile[i] = (idx.name == buf2d_last_idx) ? subgroup_size : idx.range;
  }
  ApplyTile(block, tile, false);

  auto inner = block->SubBlock(0);
  for (size_t i = 0; i < inner->idxs.size(); ++i) {
    const auto& idx = inner->idxs[i];
    tile[i] = (idx.name == buf2d_last_idx) ? (idx.range / subgroup_size) : idx.range;
  }
  ApplyTile(inner.get(), tile, false);

  auto accum = inner.get()->SubBlock(0);
  for (size_t i = 0; i < accum->idxs.size(); ++i) {
    const auto& idx = accum->idxs[i];
    tile[i] = (idx.name == buf1d_idx) ? subgroup_size : idx.range;
  }
  ApplyTile(accum.get(), tile, false);
  accum->idxs.push_back({"thread_idx", 0, Affine(sbuf_idx)});

  auto compute = accum.get()->SubBlock(0);
  auto buf1d = accum->ref_by_from(buf1d_name);
  auto sbuf = accum->ref_by_from(sbuf_name);

  // New load refinement
  std::string reg_ref_name = buf1d_name + "_reg";
  std::vector<size_t> sizes = buf1d->interior_shape.sizes();
  sizes.back() = 1;
  auto reg_shape_accum = SimpleShape(buf1d->interior_shape.type, sizes);
  reg_shape_accum.dims.emplace_back(0, subgroup_size);
  std::vector<stripe::Affine> reg_access_zero(reg_shape_accum.dims.size());
  Refinement reg_ref_accum(RefDir::None, "", reg_ref_name, reg_access_zero, reg_shape_accum, buf1d->agg_op, Location(),
                           buf1d->offset, stripe::BankDimension{reg_shape_accum.dims.size() - 1}, buf1d->cache_unit);
  accum->refs.insert(reg_ref_accum);

  // New store refinement
  std::string sreg_name = sbuf_name + "_reg";
  std::vector<stripe::Affine> sreg_access_zero(sbuf->access.size());
  Refinement sreg_ref_inner(RefDir::None, "", sreg_name, sreg_access_zero, sbuf->interior_shape, sbuf->agg_op,
                            Location(), sbuf->offset, sbuf->bank_dim, sbuf->cache_unit);
  inner->refs.insert(sreg_ref_inner);
  Refinement sreg_ref_accum(RefDir::Out, sreg_name, sreg_name, sreg_access_zero, sbuf->interior_shape, sbuf->agg_op,
                            Location(), sbuf->offset, sbuf->bank_dim, sbuf->cache_unit);
  accum->refs.insert(sreg_ref_accum);

  // New load block
  auto load_block = std::make_shared<stripe::Block>();
  load_block->name = compute->name;
  // Make thread id
  load_block->idxs.push_back({buf1d_idx, 1});
  std::string thread_idx = buf1d_idx + "_i";
  load_block->idxs.push_back({thread_idx, 0, Affine("thread_idx")});
  // Register refinement for new load block
  // new shape with subgroup dim
  auto reg_shape_load = SimpleShape(buf1d->interior_shape.type, sizes);
  reg_shape_load.dims.emplace_back(0, 1);
  // new access with subgroup dim
  std::vector<stripe::Affine> reg_access_load = reg_access_zero;
  reg_access_load.back() = Affine(buf1d_idx);
  Refinement reg_ref_load(RefDir::Out, reg_ref_name, reg_ref_name, reg_access_load, reg_shape_load,
                          reg_ref_accum.agg_op, Location(), reg_ref_accum.offset, reg_ref_accum.bank_dim,
                          reg_ref_accum.cache_unit);
  Refinement orig_ref_load = *buf1d;
  orig_ref_load.access.back() = Affine(thread_idx);
  load_block->refs.insert(reg_ref_load);
  load_block->refs.insert(orig_ref_load);
  auto load = std::make_shared<stripe::Load>(buf1d_name, "$x");
  auto store = std::make_shared<stripe::Store>("$x", reg_ref_name);
  load_block->stmts.push_back(load);
  load_block->stmts.push_back(store);
  accum->stmts.insert(accum->stmts.begin(), load_block);

  // store stmts
  auto sload = std::make_shared<stripe::Load>(sreg_name, "$" + sreg_name);
  auto sstore = std::make_shared<stripe::Store>("$" + sreg_name, sbuf_name);
  inner->stmts.push_back(sload);
  inner->stmts.push_back(sstore);

  // Modify compute block
  // Replace load refinement
  compute->refs.erase(*buf1d);
  std::vector<stripe::Affine> reg_access = reg_access_zero;
  reg_access.back().mutateMap().emplace(buf1d_idx, 1);
  Refinement reg_ref_compute(RefDir::In, reg_ref_name, buf1d_name, reg_access, reg_shape_accum, reg_ref_accum.agg_op,
                             Location(), reg_ref_accum.offset, reg_ref_accum.bank_dim, reg_ref_accum.cache_unit);
  compute->refs.insert(reg_ref_compute);
  // Replace store refinement
  compute->refs.erase(*sbuf);
  std::vector<stripe::Affine> sreg_access = sreg_access_zero;
  sreg_access.back().mutateMap().emplace(sbuf_idx, 1);
  Refinement sreg_ref_compute(RefDir::Out, sreg_name, sbuf_name, sreg_access, sbuf->interior_shape, sbuf->agg_op,
                              Location(), sbuf->offset, sbuf->bank_dim, sbuf->cache_unit);
  compute->refs.insert(sreg_ref_compute);
  // Tag load stmts
  for (auto stmt : compute->stmts) {
    auto load = Load::Downcast(stmt);
    if (load) {
      if (load->from == buf1d_name) {
        load->set_tag("subgroup_broadcast");
        load->set_tag("temp_var");
      } else if (load->from == buf2d_name) {
        load->set_attr("zero_skip", "$" + buf1d_name);
        load->set_attr("zero_error", options.zero_error());
        load->set_tag("temp_var");
      }
    }
  }
  load_block->set_tag("subgroup_inline");
  compute->set_tag("subgroup_inline");
  inner->set_tag("gpu_thread");
  inner->set_tag("subgroup_thread");
  block->remove_tag("contraction");
  block->set_attr("subgroup_size", static_cast<int64_t>(subgroup_size));
  block->set_tag("subgroup_outer");
}

void FullyConnectedPass::Apply(CompilerState* state) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs,
              [this](const AliasMap& alias_map, stripe::Block* block) {  //
                FullyConnected(alias_map, block, options_);
              },
              true);
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<FullyConnectedPass, proto::FullyConnectedPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
