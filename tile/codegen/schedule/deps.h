// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/alias.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {
namespace schedule {

// Recomputes Statement dependencies within a single Block.
void ComputeDepsForBlock(stripe::Block* block, const AliasMap& alias_map);

// Recomputes Statement dependencies within a Block, including all nested sub-Blocks.
void ComputeDepsForTree(stripe::Block* outermost_block);

}  // namespace schedule
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
