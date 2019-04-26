// Copyright 2018, Intel Corporation

#pragma once

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/deps.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void DoReorderBlocks(stripe::Block *root);

// Kernel-level block reordering
inline void ReorderBlocksPass(stripe::Block *root,
                              const proto::GenericPass &options) {
  DoReorderBlocks(root);
}

} // namespace codegen
} // namespace tile
} // namespace vertexai
