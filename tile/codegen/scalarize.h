// Copyright 2018, Intel Corp.

#pragma once

#include <string>
#include <vector>

#include "tile/codegen/tags.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void Scalarize(stripe::Block* block, bool recursive = false);

inline void ScalarizePass(stripe::Block* root, const Tags& reqs) {
  RunOnBlocks(root, reqs, [](const AliasMap& map, stripe::Block* block) { Scalarize(block, true); });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
