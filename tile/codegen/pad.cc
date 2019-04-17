// Copyright 2019, Intel Corp.

#include "tile/codegen/pad.h"

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

typedef std::map<std::string, std::vector<ExtentIO>> Extents;

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
            it->second[i].load.max = std::max(it->second[i].load.min, ai.extents[i].max);
          }
        }
      } break;
      case StmtKind::Store: {
        const auto& ai = new_map.at(Store::Downcast(stmt)->into);
        auto it = extents->find(ai.base_name);
        if (it != extents->end()) {
          for (size_t i = 0; i < ai.extents.size(); i++) {
            it->second[i].store.min = std::min(it->second[i].store.min, ai.extents[i].min);
            it->second[i].store.max = std::max(it->second[i].store.min, ai.extents[i].max);
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

void Pad(Block* block, const AliasMap& map) {
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
    ApplyCache(self, block, name, block->ref_by_into(name)->location, Location(),
               {"kernel", "eltwise", "eltwise_padding"}, {"kernel", "eltwise", "eltwise_padding"});
  }

  // Update the buffer shapes
  for (auto& ref : block->refs) {
    if (!to_pad.count(ref.into())) {
      continue;
    }
    std::string bname = self.at(ref.into()).base_name;
    const auto& exts = extents.at(bname);
    int64_t stride = 1;
    for (int i = exts.size() - 1; i >= 0; i--) {
      ref.mut().interior_shape.dims[i].stride = stride;
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
    FixupRefs(block, ref.into());
  }

  // Add the zeros
  for (const auto& name : to_pad) {
    auto zero = std::make_shared<Special>();
    zero->name = "zero";
    zero->outputs = {name};
    block->stmts.push_front(zero);
  }
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
