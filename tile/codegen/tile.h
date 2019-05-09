// Copyright 2018, Intel Corporation

#pragma once

#include <string>

#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

bool ApplyTile(stripe::Block* outer,          //
               const TileShape& shape,        //
               bool elide_trivial = true,     //
               bool copy_tags = false,        //
               bool interleave = false,       //
               bool split_unaligned = false,  //
               const std::string& location_idx_tag = "");

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
