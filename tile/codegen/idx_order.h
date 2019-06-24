// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void ReorderIndex(stripe::Block* block, const stripe::Refinement& ref);

void IdxOrder(const AliasMap& alias_map, stripe::Block* block, const proto::IdxOrderPass& options);

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
