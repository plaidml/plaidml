// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void Scalarize(stripe::Block* block, bool recursive = false);

inline void ScalarizePass(stripe::Block* root, const proto::GenericPass& options) {
  auto reqs = stripe::FromProto(options.reqs());
  RunOnBlocks(root, reqs, [](const AliasMap& map, stripe::Block* block) {  //
    Scalarize(block, true);
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
