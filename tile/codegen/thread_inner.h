// Copyright 2018, Intel Corporation

#pragma once

#include <string>

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

// Localize everything I can, don't update location (for now)
void ThreadInnerPass(const AliasMap& scope, stripe::Block* block, int64_t threads);

// Localize starting from root for things that match reqs
inline void ThreadInnerPass(stripe::Block* root, const proto::ThreadInnerPass& options) {
  auto reqs = stripe::FromProto(options.reqs());
  RunOnBlocks(root, reqs, [&options](const AliasMap& map, stripe::Block* block) {  //
    ThreadInnerPass(map, block, options.threads());
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
