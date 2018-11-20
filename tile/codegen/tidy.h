// Copyright 2018, Intel Corporation

#pragma once

#include <string>
#include <vector>

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/tags.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void PruneIndexesPass(stripe::Block* root, const proto::PruneIndexesPass& options);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
