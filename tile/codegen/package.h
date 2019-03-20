// Copyright 2019, Intel Corporation

#pragma once

#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

// Packages a block into a sub-block.
// Creates passthrough refinements for the sub-block to access.
void PackagePass(stripe::Block* root, const proto::PackagePass& options);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
