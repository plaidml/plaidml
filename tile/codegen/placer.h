// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/tags.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

// Assigns locations to all Refinements within a Block, including all
// nested sub-Blocks.  Note that all dependencies for the block and
// sub-blocks should be established when this function is called.
//
// TODO: Compute alignment correctly.
void PlaceRefinements(stripe::Block* outermost_block, std::size_t alignment = 16);

inline void MemPlacementPass(stripe::Block* root, const Tags& reqs) {
  RunOnBlocks(root, reqs, [&](const AliasMap& map, stripe::Block* block) {  //
    PlaceRefinements(block, 4);
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
