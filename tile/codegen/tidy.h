// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void PruneIndexes(stripe::Block* block, const stripe::Tags& exclude_tags);

class PruneIndexesPass final : public CompilePass {
 public:
  explicit PruneIndexesPass(const proto::PruneIndexesPass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::PruneIndexesPass options_;
};

class PruneRefinementsPass final : public CompilePass {
 public:
  explicit PruneRefinementsPass(const proto::PruneRefinementsPass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::PruneRefinementsPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
