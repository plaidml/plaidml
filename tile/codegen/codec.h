// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"

namespace vertexai {
namespace tile {
namespace codegen {

class AssignCodecPass final : public CompilePass {
 public:
  explicit AssignCodecPass(const proto::AssignCodecPass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::AssignCodecPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
