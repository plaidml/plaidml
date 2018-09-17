// Copyright 2018, Intel Corp.

#pragma once

#include <map>
#include <string>
#include <vector>

#include "tile/proto/stripe.pb.h"

namespace vertexai {
namespace tile {
namespace codegen {

void ExecuteProgram(const stripe::proto::Block& program, std::map<std::string, std::vector<float>>* buffers);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
