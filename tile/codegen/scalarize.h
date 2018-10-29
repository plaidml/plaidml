// Copyright 2018, Intel Corp.

#pragma once

#include <string>
#include <vector>

#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void Scalarize(stripe::Block* block, bool recursive = false);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
