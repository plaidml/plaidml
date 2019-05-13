// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void Scalarize(stripe::Block* block, bool recursive = false);

class ScalarizePass final : public CompilePass {
 public:
  explicit ScalarizePass(const proto::ScalarizePass& options) : options_{options} {}
  void Apply(stripe::Block* root) const final;

 private:
  proto::ScalarizePass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
