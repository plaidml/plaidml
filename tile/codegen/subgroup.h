// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void Subgroup(stripe::Block* block, const AliasMap& map, const proto::SubgroupPass& options);

inline void SubgroupPass(stripe::Block* root, const proto::SubgroupPass& options) {
  auto reqs = stripe::FromProto(options.reqs());
  RunOnBlocks(root, reqs, [&](const AliasMap& map, stripe::Block* block) {  //
    Subgroup(block, map, options);
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
