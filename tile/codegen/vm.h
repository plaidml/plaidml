// Copyright 2018, Intel Corp.

#pragma once

#include <map>
#include <string>
#include <vector>

#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void ExecuteProgram(const stripe::Block& program, std::map<std::string, std::vector<float>>* buffers);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
