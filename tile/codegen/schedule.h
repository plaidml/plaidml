// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/tags.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void ScheduleBlock(stripe::Block* block, const proto::SchedulePass& options);

// Schedules the statements within a block.
//   Creates new refinements at the block's mem_loc for its statements to access.
//     Assigns the new refinements offsets within the mem_loc.
//   Inserts IO sub-block statements as needed.
//   Updates the block's statements' dependencies for correctness.
inline void SchedulePass(stripe::Block* root, const proto::SchedulePass& options) {
  auto reqs = FromProto(options.reqs());
  RunOnBlocks(root, reqs, [&options](const AliasMap& map, stripe::Block* block) { ScheduleBlock(block, options); });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
