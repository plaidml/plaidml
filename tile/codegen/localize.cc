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
          ref.mut().interior_shape.is_const = it->interior_shape.is_const;
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
  // Set the size of the bank-dim to be 1 for simple shape purposes
  size_t orig_size = 1;
  if (it_ref->bank_dim) {
    std::swap(sizes[it_ref->bank_dim->dim_pos], orig_size);
  }
  // Change the shape
  it_ref->mut().interior_shape = SimpleShape(it_ref->interior_shape.type, sizes);
  // Fix the bankdim back up
  if (it_ref->bank_dim) {
    it_ref->mut().interior_shape.dims[it_ref->bank_dim->dim_pos].size = orig_size;
    it_ref->mut().interior_shape.dims[it_ref->bank_dim->dim_pos].stride = 0;
  }
  // Change dir + from
  it_ref->mut().dir = RefDir::None;
  it_ref->mut().from = "";
  // Clear its affines
  for (Affine& aff : it_ref->mut().access) {
    aff = 0;
  }
  // Remove any tmp tags
  it_ref->mut().remove_tag("tmp");
  // Propagate the changes
  FixupRefs(block, var_name);
}

void LocalizeBlockPass(const AliasMap& scope, Block* block, const std::set<std::string>& ref_reqs) {
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
    LocalizeBlockPass(inner_map, inner.get(), ref_reqs);
  }
}

void LocalizePass::Apply(CompilerState* state) const {
  auto reqs = stripe::FromProto(options_.reqs());
  auto ref_reqs = stripe::FromProto(options_.ref_reqs());
  RunOnBlocks(state->entry(), reqs, [&](const AliasMap& map, stripe::Block* block) {  //
    LocalizeBlockPass(map, block, ref_reqs);
  });
}

void LocateInnerBlock(Block* block, const Tags& inner_tags, const Location& loc,
                      const proto::LocateInnerBlockPass& options) {
  for (const auto& stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (!inner) {
      continue;
    }
    if (inner->has_tags(inner_tags)) {
      auto* inner_loc = &inner->location;
      if (options.append_devs()) {
        inner_loc->devs.insert(inner_loc->devs.end(), loc.devs.begin(), loc.devs.end());
      } else {
        *inner_loc = loc;
      }
    }
    LocateInnerBlock(inner.get(), inner_tags, loc, options);
  }
}

void LocateMemoryPass::Apply(CompilerState* state) const {
  auto reqs = FromProto(options_.reqs());
  auto loc = stripe::FromProto(options_.loc());
  RunOnBlocks(state->entry(), reqs, [&loc, this](const AliasMap& map, Block* block) {
    for (auto& ref : block->refs) {
      if (ref.dir == RefDir::None) {
        auto* ref_loc = &ref.mut().location;
        if (options_.append_devs()) {
          ref_loc->devs.insert(ref_loc->devs.end(), loc.devs.begin(), loc.devs.end());
        } else {
          *ref_loc = loc;
        }
        FixupRefs(block, ref.into());
      }
    }
  });
}

void LocateBlockPass::Apply(CompilerState* state) const {
  auto reqs = FromProto(options_.reqs());
  auto loc = stripe::FromProto(options_.loc());
  RunOnBlocks(state->entry(), reqs, [&loc, this](const AliasMap& map, Block* block) {  //
    auto* block_loc = &block->location;
    if (options_.append_devs()) {
      block_loc->devs.insert(block_loc->devs.end(), loc.devs.begin(), loc.devs.end());
    } else {
      *block_loc = loc;
    }
  });
}

void LocateInnerBlockPass::Apply(CompilerState* state) const {
  auto reqs = FromProto(options_.reqs());
  auto inner_reqs = FromProto(options_.inner_reqs());
  auto loc = stripe::FromProto(options_.loc());
  RunOnBlocks(state->entry(), reqs, [&](const AliasMap& map, Block* block) {  //
    LocateInnerBlock(block, inner_reqs, loc, options_);
  });
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<LocalizePass, proto::LocalizePass>::Register();
  CompilePassFactory<LocateMemoryPass, proto::LocateMemoryPass>::Register();
  CompilePassFactory<LocateBlockPass, proto::LocateBlockPass>::Register();
  CompilePassFactory<LocateInnerBlockPass, proto::LocateInnerBlockPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
