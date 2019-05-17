// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"

namespace vertexai {
namespace tile {
namespace codegen {

class RegisterCachePass final : public CompilePass {
 public:
  explicit RegisterCachePass(const proto::RegisterCachePass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::RegisterCachePass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
