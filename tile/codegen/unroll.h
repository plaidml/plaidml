// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"

namespace vertexai {
namespace tile {
namespace codegen {

class UnrollPass final : public CompilePass {
 public:
  explicit UnrollPass(const proto::UnrollPass& options) : options_{options} {}
  void Apply(stripe::Block* root) const final;

 private:
  proto::UnrollPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
