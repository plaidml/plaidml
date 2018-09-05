// Copyright 2018, Intel Corp.

#pragma once

#include "tile/lang/generate.h"
#include "tile/proto/stripe.pb.h"

namespace vertexai {
namespace tile {
namespace codegen {

void ApplyTile(stripe::proto::Block* outer, const lang::TileShape& tile);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
