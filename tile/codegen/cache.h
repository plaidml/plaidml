// Copyright 2018, Intel Corporation

#pragma once

#include <set>
#include <string>

#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void ApplyCache(stripe::Block* block,             //
                const std::string& var_name,      //
                const stripe::Location& mem_loc,  //
                const stripe::Location& xfer_loc);

void CachePass(stripe::Block* root, const proto::CachePass& options);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
