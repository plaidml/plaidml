// Copyright 2018, Intel Corp.

#pragma once

#include <map>
#include <string>

#include "tile/proto/stripe.pb.h"

namespace vertexai {
namespace tile {
namespace codegen {

void ExecuteProgram(const stripe::proto::Block& program, const std::map<std::string, float*>& buffers);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
