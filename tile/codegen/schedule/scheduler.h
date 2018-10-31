// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/schedule/deps.h"
#include "tile/codegen/schedule/placer.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {
namespace schedule {

// Schedules a tree of Blocks for execution.
inline void Reschedule(stripe::Block* block) {
  ComputeDepsForTree(block);
  PlaceRefinements(block);
}

}  // namespace schedule
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
