// Copyright 2019, Intel Corporation

#pragma once

#include "tile/codegen/compile_pass.h"

namespace vertexai {
namespace tile {
namespace codegen {

void ConvertToStripe(CompilerState* state);
void ConvertToStripeMLIR(CompilerState* state);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
