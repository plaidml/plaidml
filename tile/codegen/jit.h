// Copyright 2018, Intel Corp.

#pragma once

#include <map>
#include <string>

#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void JitExecute(const stripe::Block& program, const std::map<std::string, void*>& buffers);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
