// Copyright 2018, Intel Corporation

#pragma once

#include <set>
#include <string>

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void RegisterCachePass(stripe::Block* root, const proto::RegisterPass& options);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
