// Copyright 2018, Intel Corporation

#include "tile/codegen/localize.h"

#include <algorithm>

#include <boost/format.hpp>

#include "base/util/throw.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

void FixupRefs(Block* block, const std::string& var_name) {
  auto it = block->ref_by_into(var_name, false);
  if (it == block->refs.end()) {
    return;
  }
  for (auto stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      for (auto& ref : inner->refs) {
        if (ref.from == var_name) {
          ref.mut().location = it->location;
          ref.mut().offset = it->offset;
          for (size_t i = 0; i < ref.interior_shape.dims.size(); i++) {
            ref.mut().interior_shape.dims[i].stride = it->interior_shape.dims[i].stride;
          }
          FixupRefs(inner.get(), ref.into());
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
  for (const auto& dim : it_ref->interior_shape.dims) {
    sizes.push_back(dim.size);
  }
  // Change the shape
  it_ref->mut().interior_shape = SimpleShape(it_ref->interior_shape.type, sizes);
  // Change dir + from
  it_ref->mut().dir = RefDir::None;
  it_ref->mut().from = "";
  // Clear its affines
  for (Affine& aff : it_ref->mut().access) {
    aff = 0;
  }
  // Propagate the changes
  FixupRefs(block, var_name);
}

void LocalizePass(const AliasMap& scope, Block* block, const std::set<std::string>& ref_reqs) {
  auto use_count = scope.RefUseCounts(*block);
  for (auto& stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (!inner) {
      continue;
    }
    std::set<std::string> refs_to_localize;
    std::set<std::string> refs_to_remove;
    for (const auto& ref : inner->refs) {
      auto it = block->ref_by_into(ref.from, false);
      if (it == block->refs.end()) {
        continue;
      }
      // If this wasn't allocated in the outer block and it's not tagged with tmp, skip it for consideration
      if (it->dir != RefDir::None && !it->has_tag("tmp")) {
        continue;
      }
      // If we have a ref_req and we don't have the right tags, skip
      if (ref_reqs.size() != 0 && !ref.has_tags(ref_reqs)) {
        continue;
      }
      // If it's not uniquely located in this block, don't consider
      if (use_count[ref.from] != 1) {
        continue;
      }
      refs_to_localize.emplace(ref.into());
      refs_to_remove.emplace(ref.from);
    }
    for (const auto& name : refs_to_remove) {
      block->refs.erase(block->ref_by_into(name));
    }
    for (const auto& name : refs_to_localize) {
      LocalizeRef(inner.get(), name);
    }
    // Now localize block itself
    AliasMap inner_map(scope, inner.get());
    LocalizePass(inner_map, inner.get(), ref_reqs);
  }
}

void LocateInnerBlock(Block* block, const Tags& inner_tags, const Location& loc) {
  for (const auto& stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (!inner) {
      continue;
    }
    if (inner->has_tags(inner_tags)) {
      inner->location = loc;
    }
    LocateInnerBlock(inner.get(), inner_tags, loc);
  }
}

void LocateMemoryPass(Block* root, const proto::LocatePass& options) {
  auto reqs = FromProto(options.reqs());
  auto loc = stripe::FromProto(options.loc());
  RunOnBlocks(root, reqs, [&loc](const AliasMap& map, Block* block) {
    for (auto& ref : block->refs) {
      if (ref.dir == RefDir::None) {
        ref.mut().location = loc;
        FixupRefs(block, ref.into());
      }
    }
  });
}

void LocateBlockPass(Block* root, const proto::LocatePass& options) {
  auto reqs = FromProto(options.reqs());
  auto loc = stripe::FromProto(options.loc());
  RunOnBlocks(root, reqs, [&loc](const AliasMap& map, Block* block) {  //
    block->location = loc;
  });
}

void LocateInnerBlockPass(Block* root, const proto::LocatePass& options) {
  auto reqs = FromProto(options.reqs());
  auto inner_reqs = FromProto(options.inner_reqs());
  auto loc = stripe::FromProto(options.loc());
  RunOnBlocks(root, reqs, [&](const AliasMap& map, Block* block) {  //
    LocateInnerBlock(block, inner_reqs, loc);
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
