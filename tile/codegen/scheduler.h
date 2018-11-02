// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/deps.h"
#include "tile/codegen/placer.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

// Schedules a tree of Blocks for execution.
inline void Reschedule(stripe::Block* block) {
  ComputeDepsForTree(block);
  PlaceRefinements(block);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
