// Copyright 2018, Intel Corporation

#pragma once

#include <map>
#include <set>
#include <vector>

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void DeadCodeElimination(const AliasMap& alias_map, stripe::Block* block);

inline void DeadCodeEliminationPass(stripe::Block* root, const proto::GenericPass& options) {
  auto reqs = stripe::FromProto(options.reqs());
  RunOnBlocks(root, reqs,
              [](const AliasMap& alias_map, stripe::Block* block) { DeadCodeElimination(alias_map, block); }, true);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
