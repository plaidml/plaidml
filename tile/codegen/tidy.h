// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void PruneIndexes(stripe::Block* block);
void PruneIndexesPass(stripe::Block* root, const proto::GenericPass& options);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
