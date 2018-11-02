// Copyright 2018, Intel Corp.

#pragma once

#include <string>
#include <vector>

#include "tile/codegen/alias.h"
#include "tile/codegen/tags.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

// Fixup all downstream reference to var_name (the into name inside of block)
// Specifically, move the location down, as well as the strides
void FixupRefs(stripe::Block* block, const std::string& var_name);

// Make var_name a local and restride to match size
// Also, propagate this on down
void LocalizeRef(stripe::Block* block, const std::string& var_name);

// Localize everything I can, don't update location (for now)
void LocalizePass(const AliasMap& scope, stripe::Block* block);

// Localize starting from root for things that match reqs
inline void LocalizePass(stripe::Block* root, const Tags& reqs) {
  RunOnBlocks(root, reqs, [](const AliasMap& map, stripe::Block* block) { LocalizePass(map, block); });
}

struct LocateMemoryPassOptions {
  Tags reqs;
  stripe::Location location;
};

void LocateMemoryPass(stripe::Block* root, const LocateMemoryPassOptions& options);
void LocateBlockPass(stripe::Block* root, const LocateMemoryPassOptions& options);
void LocateInnerBlockPass(stripe::Block* root, const LocateMemoryPassOptions& options);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
