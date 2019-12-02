// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

bool IsGlobalRef(const stripe::Refinement& ref);

bool IsRegisterRef(const stripe::Refinement& ref);

// Reorder the index in block
// global_only: Take only global refinements into account
// apply_inner: Assume block is tiled. Order the index in the inner block
void ReorderIndex(stripe::Block* block, bool global_only, bool apply_inner);

class IdxOrderPass final : public CompilePass {
 public:
  explicit IdxOrderPass(const proto::IdxOrderPass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::IdxOrderPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
