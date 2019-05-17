// Copyright 2018, Intel Corporation

#pragma once

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

// Recomputes Statement dependencies within a single Block.
void ComputeDepsForBlock(stripe::Block* block, const AliasMap& alias_map);

class ComputeDepsPass final : public CompilePass {
 public:
  explicit ComputeDepsPass(const proto::ComputeDepsPass& options) : options_{options} {}
  void Apply(stripe::Block* root) const final;

 private:
  proto::ComputeDepsPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
