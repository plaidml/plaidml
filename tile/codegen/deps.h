// Copyright 2018, Intel Corporation

#pragma once

#include <memory>

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

// Determine if the stmt is a zero block
inline bool ZeroBlock(const std::shared_ptr<stripe::Statement>& stmt) {
  auto block = stripe::Block::Downcast(stmt);
  if (block && block->has_tag("zero")) {
    return true;
  }
  auto special = stripe::Special::Downcast(stmt);
  if (special && special->name == "zero") {
    return true;
  }
  return false;
}

// Recomputes Statement dataflow dependencies within a single Block.
//
// After this call, each statement X's dependencies will be the set of all statements that write to an input
// of X -- the dataflow dependencies.  E.g. if A's dataflow dependencies are B and C, and B also depends on C,
// A's statement dependencies will be [B, C].
void ComputeDataflowDepsForBlock(stripe::Block* block, const AliasMap& alias_map);

// Recomputes Statement dependencies within a single Block.
//
// After this call, each statement X's dependencies will be a set of statements that must be completed in
// order for X's inputs to be ready.  E.g. if A's dataflow dependencies are B and C, and B also depends on C,
// A's statement dependencies will be [B].
void ComputeDepsForBlock(stripe::Block* block, const AliasMap& alias_map);

class ComputeDepsPass final : public CompilePass {
 public:
  explicit ComputeDepsPass(const proto::ComputeDepsPass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::ComputeDepsPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
