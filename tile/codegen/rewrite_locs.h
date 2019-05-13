// Copyright 2019, Intel Corporation

#pragma once

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"

namespace vertexai {
namespace tile {
namespace codegen {

class RewriteLocationsPass final : public CompilePass {
 public:
  explicit RewriteLocationsPass(const proto::RewriteLocationsPass& options) : options_{options} {}
  void Apply(stripe::Block* root) const final;

 private:
  proto::RewriteLocationsPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
