// Copyright 2018, Intel Corporation

#pragma once

#include <string>

#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

std::string EmitC(const stripe::Block& program);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
