// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"

namespace vertexai {
namespace tile {
namespace codegen {

class AutotilePass final : public CompilePass {
 public:
  explicit AutotilePass(const proto::AutotilePass& options) : options_{options} {}
  void Apply(stripe::Block* root) const final;

 private:
  proto::AutotilePass options_;
};

class PartitionComputePass final : public CompilePass {
 public:
  explicit PartitionComputePass(const proto::PartitionComputePass& options) : options_{options} {}
  void Apply(stripe::Block* root) const final;

 private:
  proto::PartitionComputePass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
