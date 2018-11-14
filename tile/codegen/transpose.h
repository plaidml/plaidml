// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/tags.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void TransposePass(stripe::Block* root, const proto::TransposePass& options);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
