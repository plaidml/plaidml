// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void Subgroup(stripe::Block* block, const AliasMap& map, const proto::SubgroupPass& options);

void VectorizeTx(stripe::Block* block, const AliasMap& map, size_t read_align_bytes, size_t write_align_bytes);

class SubgroupPass final : public CompilePass {
 public:
  explicit SubgroupPass(const proto::SubgroupPass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::SubgroupPass options_;
};

class VectorizePass final : public CompilePass {
 public:
  explicit VectorizePass(const proto::VectorizePass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::VectorizePass options_;
};
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
