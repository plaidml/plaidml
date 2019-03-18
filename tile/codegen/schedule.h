// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void ScheduleBlock(const AliasMap& alias_map, stripe::Block* block, const proto::SchedulePass& options);

// Schedules the statements within a block.
//   Creates new refinements at the block's mem_loc for its statements to access.
//     Assigns the new refinements offsets within the mem_loc.
//   Inserts IO sub-block statements as needed.
//   Updates the block's statements' dependencies for correctness.
inline void SchedulePass(stripe::Block* root, const proto::SchedulePass& options) {
  // SchedulePass assumes stmt->deps is empty here.
  // However, it may not be ture due to the previous passes.
  // So we need to enforce it here.
  stripe::Tags reqs_all;
  reqs_all.insert("all");
  RunOnBlocks(root, reqs_all,
              [](const AliasMap& map, stripe::Block* block) {  //
                for (auto& stmt : block->stmts) {
                  stmt.get()->deps.clear();
                }
              },
              true);

  auto reqs = stripe::FromProto(options.reqs());
  RunOnBlocks(root, reqs, [&options](const AliasMap& map, stripe::Block* block) {  //
    ScheduleBlock(map, block, options);
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
