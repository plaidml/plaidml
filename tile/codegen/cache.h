// Copyright 2018, Intel Corporation

#pragma once

#include <set>
#include <string>

#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void CachePass(stripe::Block* root, const proto::CachePass& options);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
