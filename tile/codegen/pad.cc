// Copyright 2019, Intel Corp.

#include "tile/codegen/pad.h"

#include <algorithm>
#include <memory>
#include <set>
#include <vector>

#include "base/util/any_factory_map.h"
#include "tile/codegen/cache.h"
#include "tile/codegen/localize.h"
#include "tile/math/util.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

struct ExtentIO {
  explicit ExtentIO(int64_t init) : load{init, init}, store{init, init} {}
  Extent load;
  Extent store;
};

typedef std::map<std::string, std::vector<uint64_t>> RefSize;
typedef std::map<std::string, Refinement> RefMap;
typedef std::map<std::string, std::vector<ExtentIO>> Extents;

static std::set<std::string> special_set = {"reshape"};

void CollectRefDefine(stripe::Block* block, RefDefineMap* ref_def_map) {
  for (auto it = block->stmts.begin(); it != block->stmts.end(); ++it) {
    auto& stmt = *it;
    if (stmt->kind() == StmtKind::Special) {
      auto special = Special::Downcast(stmt);
      if (special_set.find(special->name) != special_set.end()) {
        auto ref_it = block->ref_by_into(special->outputs[0]);
        ref_it->mut().set_tag(special->name);
        ref_it->mut().set_tag("rewrite");
        RefDefine ref_def = {ref_it->into(), block, it};
        ref_def_map->emplace(ref_it->into(), ref_def);
      }
    }
  }
}

static void CopyRefinement(const RefDefineMap& ref_def_map,  //
                           const Refinement& old_ref,        //
                           const Refinement& new_ref,        //
                           const std::vector<uint64_t>& pad_size) {
  auto block = std::make_shared<Block>();
  size_t n_dim = new_ref.interior_shape.dims.size();
  Refinement src_ref(RefDir::In, old_ref.into(), "src", old_ref.access, old_ref.interior_shape, "", old_ref.location,
                     old_ref.offset, old_ref.bank_dim, old_ref.cache_unit);
  Refinement dst_ref(RefDir::Out, new_ref.into(), "dst", new_ref.access, new_ref.interior_shape, "", new_ref.location,
                     new_ref.offset, new_ref.bank_dim, new_ref.cache_unit);

  for (size_t i = 0; i < n_dim; ++i) {
    std::string idx_name = "i" + std::to_string(i);
    block->idxs.push_back({idx_name, old_ref.interior_shape.dims[i].size, Affine()});
    src_ref.access[i] = Affine(idx_name);
    dst_ref.access[i] = Affine(idx_name) + Affine(pad_size[i]);
  }
  block->refs.insert(src_ref);
  block->refs.insert(dst_ref);

  auto load = std::make_shared<Load>("src", "$X");
  block->stmts.push_back(load);
  auto store = std::make_shared<Store>("$X", "dst");
  block->stmts.push_back(store);

  const RefDefine& ref_def = ref_def_map.at(new_ref.into());
  auto after = ref_def.stmt_iter;
  block->set_tag("kernel");
  block->name = "kernel_" + std::to_string(ref_def.block->stmts.size()) + "(" + old_ref.into() + ")";
  ref_def.block->stmts.insert(++after, block);
}

void ComputeExtents(Block* block, const AliasMap& map, Extents* extents) {
  AliasMap new_map(map, block);
  for (auto stmt : block->stmts) {
    switch (stmt->kind()) {
      case StmtKind::Load: {
        const auto& ai = new_map.at(Load::Downcast(stmt)->from);
        auto it = extents->find(ai.base_name);
        if (it != extents->end()) {
          for (size_t i = 0; i < ai.extents.size(); i++) {
            it->second[i].load.min = std::min(it->second[i].load.min, ai.extents[i].min);
            it->second[i].load.max = std::max(it->second[i].load.max, ai.extents[i].max);
          }
        }
      } break;
      case StmtKind::Store: {
        const auto& ai = new_map.at(Store::Downcast(stmt)->into);
        auto it = extents->find(ai.base_name);
        if (it != extents->end()) {
          for (size_t i = 0; i < ai.extents.size(); i++) {
            it->second[i].store.min = std::min(it->second[i].store.min, ai.extents[i].min);
            it->second[i].store.max = std::max(it->second[i].store.max, ai.extents[i].max);
          }
        }
      } break;
      case StmtKind::Block:
        ComputeExtents(Block::Downcast(stmt).get(), map, extents);
        break;
      default:
        break;
    }
  }
}

// Given the new index for a block
// 1) Set the new size for the block's output
// 2) Add a new block for transfering the new output to the original output
void ModifyBlockIdxs(Block* block, const std::map<std::string, size_t>& new_idxs, Block* parent, StatementIt stmt_it) {
  // Check if the output refs are affected
  std::set<std::string> affected_out;
  for (const auto& ref : block->ref_outs()) {
    for (const auto& acc : ref->access) {
      const auto acc_map = acc.getMap();
      for (const auto it : new_idxs) {
        if (acc_map.find(it.first) != acc_map.end()) {
          if (acc_map.size() > 1) {
            // If the access is affected by the new idx
            // and the access is complex, give up
            return;
          }
          affected_out.insert(ref->from);
        }
      }
    }
  }
  // For each affected ref, we have to generate a new block for reshape
  for (const auto& ref_name : affected_out) {
    auto reshape = std::make_shared<Block>();
    auto parent_ref_it = parent->ref_by_into(ref_name);
    auto block_ref_it = block->ref_by_from(ref_name);
    std::vector<Affine> access;
    size_t n_dim = parent_ref_it->interior_shape.dims.size();
    std::vector<size_t> src_shape_dims;
    for (size_t i = 0; i < n_dim; ++i) {
      const auto& affine = block_ref_it->access[i];
      const auto& acc_map = affine.getMap();
      if (acc_map.size() == 1) {
        const auto& idx_name = acc_map.begin()->first;
        auto it = new_idxs.find(idx_name);
        if (it != new_idxs.end()) {
          src_shape_dims.push_back(it->second);
        } else {
          src_shape_dims.push_back(parent_ref_it->interior_shape.dims[i].size);
        }
      } else {
        src_shape_dims.push_back(parent_ref_it->interior_shape.dims[i].size);
      }
    }
    TensorShape src_outer_shape = SimpleShape(parent_ref_it->interior_shape.type, src_shape_dims);
    TensorShape src_inner_shape = src_outer_shape;
    TensorShape dst_inner_shape = parent_ref_it->interior_shape;
    for (size_t i = 0; i < n_dim; ++i) {
      std::string idx_name = "i" + std::to_string(i);
      reshape->idxs.push_back({idx_name, parent_ref_it->interior_shape.dims[i].size});
      access.push_back(Affine(idx_name));
      src_inner_shape.dims[i].size = 1;
      dst_inner_shape.dims[i].size = 1;
    }
    std::string src_ref_name = ref_name + "_copy";
    Refinement src_outer_ref(RefDir::None, "", src_ref_name, parent_ref_it->access, src_outer_shape,
                             parent_ref_it->agg_op, parent_ref_it->location, parent_ref_it->offset,
                             parent_ref_it->bank_dim, parent_ref_it->cache_unit);
    parent->refs.insert(src_outer_ref);
    Refinement src_inner_ref(RefDir::In, src_ref_name, src_ref_name, access, src_inner_shape, parent_ref_it->agg_op,
                             parent_ref_it->location, parent_ref_it->offset, parent_ref_it->bank_dim,
                             parent_ref_it->cache_unit);
    reshape->refs.insert(src_inner_ref);
    Refinement dst_inner_ref(RefDir::Out, ref_name, ref_name, access, dst_inner_shape, parent_ref_it->agg_op,
                             parent_ref_it->location, parent_ref_it->offset, parent_ref_it->bank_dim,
                             parent_ref_it->cache_unit);
    reshape->refs.insert(dst_inner_ref);
    auto load = std::make_shared<Load>(src_ref_name, "$x");
    auto store = std::make_shared<Store>("$x", ref_name);
    reshape->stmts.push_back(load);
    reshape->stmts.push_back(store);
    reshape->set_tag("eltwise");
    reshape->set_tag("kernel");
    reshape->name = "kernel_" + std::to_string(parent->stmts.size()) + "(" + src_ref_name + ")";
    // Modify the ref in the original block
    Refinement new_block_ref(RefDir::Out, src_ref_name, block_ref_it->into(), block_ref_it->access, src_inner_shape,
                             block_ref_it->agg_op, block_ref_it->location, block_ref_it->offset, block_ref_it->bank_dim,
                             block_ref_it->cache_unit);
    block->refs.erase(*block_ref_it);
    block->refs.insert(new_block_ref);
    for (auto& idx : block->idxs) {
      auto it = new_idxs.find(idx.name);
      if (it != new_idxs.end()) {
        idx.range = it->second;
      }
    }
    // Add reshape to parent
    parent->stmts.insert(++stmt_it, reshape);
  }
}

bool QualifiedBlock(Block* block) {
  return block && block->has_tag("agg_op_add") &&
         (block->has_tag("combo_op_mul") || block->has_tag("agg_op_add_no_combo_op"));
}

// If the dimension's range is a large prime,
// change it to a better number that can be divisible by more factors
void PrimeDimension(Block* block, const proto::PadPass& options) {
  // Check if the idxs are large primes, and the output accesses are simple
  for (StatementIt stmt_it = block->stmts.begin(); stmt_it != block->stmts.end(); ++stmt_it) {
    auto inner = Block::Downcast(*stmt_it);
    if (QualifiedBlock(inner.get())) {
      std::map<std::string, size_t> new_idxs;
      for (auto& idx : inner->idxs) {
        if (idx.range > options.prime_threshold() && math::IsPrime(idx.range)) {
          new_idxs.emplace(idx.name, (idx.range / 8 + 1) * 8);
        }
      }
      if (new_idxs.size() > 0) {
        ModifyBlockIdxs(inner.get(), new_idxs, block, stmt_it);
      }
    }
  }
}

void Pad(Block* block, const AliasMap& map, const RefDefineMap& ref_def_map) {
  // Generate a map extents for possible padding candidates
  Extents extents;
  // Look for buffers that are used for multiply accumulates
  for (auto stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (QualifiedBlock(inner.get())) {
      // Add any inputs as possible candidates
      for (auto ref : inner->ref_ins()) {
        std::string bname = map.at(ref->from).base_name;
        if (extents.count(bname)) {
          continue;
        }
        for (const auto& a : ref->access) {
          extents[bname].emplace_back(a.constant());
        }
      }
      ComputeExtents(inner.get(), map, &extents);
    }
  }
  // Now decide which ones we will be padding, and which need caching
  std::set<std::string> to_pad;
  std::set<std::string> to_cache;
  for (auto& ref : block->refs) {
    std::string bname = map.at(ref.into()).base_name;
    auto it = extents.find(bname);
    if (it == extents.end()) {
      continue;
    }
    const auto& exts = it->second;
    for (size_t i = 0; i < exts.size(); i++) {
      if (exts[i].load.min < 0 || exts[i].load.max >= static_cast<int64_t>(ref.interior_shape.dims[i].size)) {
        to_pad.insert(ref.into());
        if (ref.dir != RefDir::None) {
          to_cache.insert(ref.into());
        }
      }
    }
  }

  // Remove constraints that will no longer be required
  for (auto stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    std::vector<Affine> new_cons;
    if (QualifiedBlock(inner.get())) {
      for (const auto& con : inner->constraints) {
        bool is_safe = false;
        // Remove all constraints that are simple edge checks on inputs.
        // Presuming that constraint was the only thing preventing this block
        // from executing, the block will now read what used to be out of
        // bounds, but is now a zeroed value.  Since 0 values for multiply
        // accumulate are identity, the removing the constraint is safe
        for (auto ref : inner->ref_ins()) {
          for (size_t i = 0; i < ref->access.size(); i++) {
            const auto& ai = map.at(ref->from);
            // Check if constraint is a lower bound match
            if (ref->access[i] - con == Affine()) {
              is_safe = true;
            }
            // Check if constraint is an upper bound match
            if (ref->access[i] + con == Affine(ai.base_ref->interior_shape.dims[i].size - 1)) {
              is_safe = true;
            }
          }
        }
        if (!is_safe) {
          // Keep constraints that are not safe to remove
          new_cons.push_back(con);
        }
      }
      inner->constraints = new_cons;
    }
  }

  // Do the caching
  for (const auto& name : to_cache) {
    auto ref_it = block->ref_by_into(name);
    Location loc = ref_it->location;
    ApplySimpleCache(map, IsWriteDir(ref_it->dir) ? RefDir::Out : RefDir::In, block, name, loc, Location(),
                     {"kernel", "eltwise", "eltwise_padding"}, {"kernel", "eltwise", "eltwise_padding"});
  }

  RefSize pad_sizes;
  RefMap old_refs;

  // Update the buffer shapes
  for (auto& ref : block->refs) {
    if (!to_pad.count(ref.into())) {
      continue;
    }
    std::vector<uint64_t> pad_size;
    old_refs.emplace(ref.into(), ref);
    std::string bname = map.at(ref.into()).base_name;
    const auto& exts = extents.at(bname);
    int64_t stride = 1;
    for (int i = exts.size() - 1; i >= 0; i--) {
      ref.mut().interior_shape.dims[i].stride = stride;
      // When padding the new buffer should be bigger and there should not be negative offsets.
      int64_t padSize = -exts[i].load.min;
      if (padSize < 0) {
        padSize = 0;
      }
      pad_size.push_back(padSize);
      uint64_t new_size = exts[i].load.max + 1 - exts[i].load.min;
      // N.B. Adding padSize to the interior_shape.size keeps the load block within bounds.
      new_size = std::max(new_size, ref.interior_shape.dims[i].size + padSize);
      ref.mut().interior_shape.dims[i].size = new_size;
      stride *= new_size;
      // Bump all the interior pointers!
      for (auto stmt : block->stmts) {
        auto inner = stripe::Block::Downcast(stmt);
        if (!inner) {
          continue;
        }
        for (auto& refi : inner->refs) {
          if (refi.from == ref.into()) {
            refi.mut().access[i] += padSize;
          }
        }
      }
    }
    std::reverse(pad_size.begin(), pad_size.end());
    pad_sizes.emplace(ref.into(), pad_size);
    FixupRefs(block, ref.into());
  }

  for (const auto& name : to_pad) {
    auto ref_it = block->ref_by_into(name);
    auto& ref = *ref_it;
    if (ref.has_tag("rewrite")) {
      auto& ref_def = ref_def_map.at(name);
      auto& stmt = *(ref_def.stmt_iter);
      auto special = Special::Downcast(stmt);
      std::string tmp_ref_name;
      if (special->has_tag("replica")) {
        tmp_ref_name = special->outputs[0];
      } else {
        tmp_ref_name = block->unique_ref_name(name);
        special->outputs[0] = tmp_ref_name;
        special->set_tag("replica");
        auto& old_ref = old_refs.at(name);
        Refinement tmp_ref(old_ref.dir, "", tmp_ref_name, old_ref.access, old_ref.interior_shape, "", old_ref.location,
                           old_ref.offset, old_ref.bank_dim, old_ref.cache_unit);
        tmp_ref.set_tag("rewrite");
        ref_def.block->refs.insert(tmp_ref);
      }
      auto old_ref_it = ref_def.block->ref_by_into(tmp_ref_name);
      CopyRefinement(ref_def_map, *old_ref_it, *ref_it, pad_sizes.at(name));
    }
  }

  // Add the zeros
  for (const auto& name : to_pad) {
    auto zero = std::make_shared<Special>();
    zero->name = "zero";
    zero->outputs = {name};
    block->stmts.push_front(zero);
  }
  // For the Out refs initialized by zero, set them as InOut
  for (auto stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (QualifiedBlock(inner.get())) {
      for (auto ref : inner->ref_outs()) {
        if (to_pad.find(ref->from) != to_pad.end()) {
          ref->mut().dir = RefDir::InOut;
          ref->mut().set_tag("initialized");
        }
      }
    }
  }
}

void PadPass::Apply(CompilerState* state) const {
  stripe::Block* root = state->entry();
  auto reqs = stripe::FromProto(options_.reqs());
  RefDefineMap ref_def_map;
  PrimeDimension(root->SubBlock(0).get(), options_);
  CollectRefDefine(root->SubBlock(0).get(), &ref_def_map);
  RunOnBlocks(state->entry(), reqs, [&](const AliasMap& map, stripe::Block* block) {  //
    Pad(block, map, ref_def_map);
  });
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<PadPass, proto::PadPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
