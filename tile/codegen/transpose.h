// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"

namespace vertexai {
namespace tile {
namespace codegen {

class TransposePass final : public CompilePass {
 public:
  explicit TransposePass(const proto::TransposePass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::TransposePass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
