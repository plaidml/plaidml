// Copyright 2018, Intel Corporation

#pragma once

#include <string>
#include <vector>

#include <boost/optional.hpp>

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/tile.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void AutotilePass(stripe::Block* block, const proto::AutotilePass& options);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
