// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"

namespace vertexai {
namespace tile {
namespace codegen {

class ReorderBlocksPass final : public CompilePass {
 public:
  explicit ReorderBlocksPass(const proto::ReorderBlocksPass& options) : options_{options} {}
  void Apply(stripe::Block* root) const final;

 private:
  proto::ReorderBlocksPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
