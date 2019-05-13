// Copyright 2019, Intel Corporation

#pragma once

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"

namespace vertexai {
namespace tile {
namespace codegen {

class MemRebasePass final : public CompilePass {
 public:
  explicit MemRebasePass(const proto::MemRebasePass& options) : options_{options} {}
  void Apply(stripe::Block* root) const final;

 private:
  proto::MemRebasePass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
