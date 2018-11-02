// Copyright 2018, Intel Corp.

#include "tile/codegen/localize.h"

#include <algorithm>

#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

void FixupRefs(Block* block, const std::string& var_name) {
  auto ref_it = block->ref_by_into(var_name);
  if (ref_it == block->refs.end()) {
    return;
  }
  for (auto stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      for (auto& ref : inner->refs) {
        if (ref.from == var_name) {
          ref.location = ref_it->location;
          for (size_t i = 0; i < ref.shape.dims.size(); i++) {
            ref.shape.dims[i].stride = ref_it->shape.dims[i].stride;
          }
          FixupRefs(inner.get(), ref.into);
        }
      }
    }
  }
}

// Make var_name a local (in location) and restride to match size
// Also, propagate this on down
void LocalizeRef(Block* block, const std::string& var_name) {
  auto it_ref = block->ref_by_into(var_name);
  // Get the sizes
  std::vector<size_t> sizes;
  for (const auto& dim : it_ref->shape.dims) {
    sizes.push_back(dim.size);
  }
  // Change the shape
  it_ref->shape = SimpleShape(it_ref->shape.type, sizes);
  // Change dir + from
  it_ref->dir = RefDir::None;
  it_ref->from = "";
  // Clear its affines
  for (auto& aff : it_ref->access) {
    aff = 0;
  }
  // Propagate the changes
  FixupRefs(block, var_name);
}

void LocalizePass(const AliasMap& scope, Block* block) {
  // Compute statement use count of each buffer
  std::map<std::string, size_t> use_count;
  for (const auto& stmt : block->stmts) {
    std::set<std::string> buf_use;
    for (const auto& str : stmt->buffer_reads()) {
      buf_use.emplace(scope.at(str).base_name);
    }
    for (const auto& str : stmt->buffer_writes()) {
      buf_use.emplace(scope.at(str).base_name);
    }
    for (const auto& str : buf_use) {
      use_count[str]++;
    }
  }
  for (auto& stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (!inner) {
      continue;
    }
    std::set<std::string> refs_to_localize;
    std::set<std::string> refs_to_remove;
    for (const auto& ref : inner->refs) {
      // If ref not defined in the outer block, don't consider
      if (block->ref_by_into(ref.from)->dir != RefDir::None) {
        continue;
      }
      // If it's not uniquely located in this block, don't consider
      if (use_count[scope.at(ref.from).base_name] != 1) {
        continue;
      }
      refs_to_localize.emplace(ref.into);
      refs_to_remove.emplace(ref.from);
    }
    for (const auto& name : refs_to_remove) {
      block->refs.erase(block->ref_by_into(name));
    }
    for (const auto& name : refs_to_localize) {
      LocalizeRef(inner.get(), name);
    }
    // Now localize block itself
    AliasMap inner_map(scope, *inner);
    LocalizePass(inner_map, inner.get());
  }
}

void RecursiveLocate(Block* block, Location location) {
  for (const auto& stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (!inner) {
      continue;
    }
    inner->location = location;
    RecursiveLocate(inner.get(), location);
  }
}

void LocateMemoryPass(Block* root, const LocateMemoryPassOptions& options) {
  RunOnBlocks(root, options.reqs, [&](const AliasMap& map, Block* block) {
    for (auto& ref : block->refs) {
      if (ref.dir == RefDir::None) {
        ref.location = options.location;
        FixupRefs(block, ref.into);
      }
    }
  });
}

void LocateBlockPass(Block* root, const LocateMemoryPassOptions& options) {
  RunOnBlocks(root, options.reqs, [&](const AliasMap& map, Block* block) {  //
    block->location = options.location;
  });
}

void LocateInnerBlockPass(Block* root, const LocateMemoryPassOptions& options) {
  RunOnBlocks(root, options.reqs, [&](const AliasMap& map, Block* block) {  //
    RecursiveLocate(block, options.location);
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
