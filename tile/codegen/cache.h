// Copyright 2018, Intel Corp.

#pragma once

#include <string>
#include <vector>

#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void ApplyCache(stripe::Block* block, const std::string& var_name);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
