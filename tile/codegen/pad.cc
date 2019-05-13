// Copyright 2019, Intel Corp.

#include "tile/codegen/pad.h"

#include "base/util/any_factory_map.h"
#include "tile/codegen/cache.h"
#include "tile/codegen/localize.h"

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

void Pad(Block* block, const AliasMap& map, const RefDefineMap& ref_def_map) {
  AliasMap self(map, block);
  // Generate a map extents for possible padding candidates
  Extents extents;
  // Look for buffers that are used for multiply accumulates
  for (auto stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner && inner->has_tag("agg_op_add") && inner->has_tag("comb_op_mul")) {
      // Add any inputs as possible candidates
      for (auto ref : inner->ref_ins()) {
        std::string bname = self.at(ref->from).base_name;
        if (extents.count(bname)) {
          continue;
        }
        for (const auto& a : ref->access) {
          extents[bname].emplace_back(a.constant());
        }
      }
      ComputeExtents(inner.get(), self, &extents);
    }
  }
  // Now decide which ones we will be padding, and which need caching
  std::set<std::string> to_pad;
  std::set<std::string> to_cache;
  for (auto& ref : block->refs) {
    std::string bname = self.at(ref.into()).base_name;
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
    if (inner && inner->has_tag("agg_op_add") && inner->has_tag("comb_op_mul")) {
      for (const auto& con : inner->constraints) {
        bool is_safe = false;
        // Remove all constraints that are simple edge checks on inputs.
        // Presuming that constraint was the only thing preventing this block
        // from executing, the block will now read what used to be out of
        // bounds, but is now a zeroed value.  Since 0 values for multiply
        // accumulate are identity, the removing the constraint is safe
        for (auto ref : inner->ref_ins()) {
          for (size_t i = 0; i < ref->access.size(); i++) {
            const auto& ai = self.at(ref->from);
            // Check if constraint is a lower bound match
            if (ref->access[i] - con == Affine()) {
              // IVLOG(1, "Lower bound: " << con);
              is_safe = true;
            }
            // Check if constraint is an upper bound match
            if (ref->access[i] + con == Affine(ai.base_ref->interior_shape.dims[i].size - 1)) {
              // IVLOG(1, "Upper bound: " << con);
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
    // IVLOG(1, "Cacheing: " << name);
    Location loc = block->ref_by_into(name)->location;
    ApplyCache(self, block, name, loc, Location(), {"kernel", "eltwise", "eltwise_padding"},
               {"kernel", "eltwise", "eltwise_padding"});
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
    std::string bname = self.at(ref.into()).base_name;
    const auto& exts = extents.at(bname);
    int64_t stride = 1;
    for (int i = exts.size() - 1; i >= 0; i--) {
      ref.mut().interior_shape.dims[i].stride = stride;
      pad_size.push_back(-exts[i].load.min);
      uint64_t new_size = exts[i].load.max + 1 - exts[i].load.min;
      new_size = std::max(new_size, ref.interior_shape.dims[i].size);
      ref.mut().interior_shape.dims[i].size = new_size;
      stride *= new_size;
      // Bump all the interior pointers!
      for (auto stmt : block->stmts) {
        auto inner = stripe::Block::Downcast(stmt);
        if (!inner) continue;
        for (auto& refi : inner->refs) {
          if (refi.from == ref.into()) {
            refi.mut().access[i] += -exts[i].load.min;
          }
        }
      }
      // ref.mut().access[i] += -exts[i].load.min;
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
}

void PadPass::Apply(stripe::Block* root) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RefDefineMap ref_def_map;
  CollectRefDefine(root->SubBlock(0).get(), &ref_def_map);
  RunOnBlocks(root, reqs, [&](const AliasMap& map, stripe::Block* block) {  //
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
