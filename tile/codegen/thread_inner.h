// Copyright 2018, Intel Corporation

#pragma once

#include <string>

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

// Localize everything I can, don't update location (for now)
void DoThreadInnerPass(const AliasMap& scope, stripe::Block* block, int64_t threads);

class ThreadInnerPass final : public CompilePass {
 public:
  explicit ThreadInnerPass(const proto::ThreadInnerPass& options) : options_{options} {}
  void Apply(stripe::Block* root) const final;

 private:
  proto::ThreadInnerPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
