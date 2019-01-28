// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

// Assigns locations to all Refinements within a Block, including all
// nested sub-Blocks.  Note that all dependencies for the block and
// sub-blocks should be established when this function is called.
void PlaceRefinements(stripe::Block* outermost_block, const proto::MemoryPlacementPass& options);

inline void MemPlacementPass(stripe::Block* root, const proto::MemoryPlacementPass& options) {
  auto reqs = stripe::FromProto(options.reqs());
  RunOnBlocks(root, reqs, [&](const AliasMap& map, stripe::Block* block) {  //
    PlaceRefinements(block, options);
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
