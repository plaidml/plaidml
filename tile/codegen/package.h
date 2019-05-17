// Copyright 2019, Intel Corporation

#pragma once

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"

namespace vertexai {
namespace tile {
namespace codegen {

// Packages a block into a sub-block.
// Creates passthrough refinements for the sub-block to access.
class PackagePass final : public CompilePass {
 public:
  explicit PackagePass(const proto::PackagePass& options) : options_{options} {}
  void Apply(stripe::Block* root) const final;

 private:
  proto::PackagePass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
