// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/alias.h"
#include "tile/codegen/tags.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

// Recomputes Statement dependencies within a single Block.
void ComputeDepsForBlock(stripe::Block* block, const AliasMap& alias_map);

// Recomputes Statement dependencies within a Block, including all nested sub-Blocks.
void ComputeDepsForTree(stripe::Block* outermost_block);

inline void ComputeDepsPass(stripe::Block* root, const Tags& reqs) {
  RunOnBlocks(root, reqs, [&](const AliasMap& map, stripe::Block* block) {  //
    ComputeDepsForBlock(block, map);
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
