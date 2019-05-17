// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

// Assigns locations to all Refinements within a Block, including all
// nested sub-Blocks.  Note that all dependencies for the block and
// sub-blocks should be established when this function is called.
void PlaceRefinements(stripe::Block* outermost_block, const proto::MemoryPlacementPass& options);

class MemoryPlacementPass final : public CompilePass {
 public:
  explicit MemoryPlacementPass(const proto::MemoryPlacementPass& options) : options_{options} {}
  void Apply(stripe::Block* root) const final;

 private:
  proto::MemoryPlacementPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
