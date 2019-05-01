// Copyright 2019, Intel Corporation

#pragma once

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void MemRebasePass(stripe::Block* root, const proto::MemRebasePass& options);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
