// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

// Recomputes Statement dependencies within a single Block.
void ComputeDepsForBlock(stripe::Block* block, const AliasMap& alias_map);

// Recomputes Statement dependencies within all matching Blocks.
inline void ComputeDepsPass(stripe::Block* root, const proto::GenericPass& options) {
  auto reqs = stripe::FromProto(options.reqs());
  RunOnBlocks(root, reqs,
              [](const AliasMap& map, stripe::Block* block) {  //
                ComputeDepsForBlock(block, map);
              },
              true);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
