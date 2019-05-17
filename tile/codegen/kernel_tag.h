// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void KernelTag(const AliasMap& alias_map, stripe::Block* block);

class KernelTagPass final : public CompilePass {
 public:
  explicit KernelTagPass(const proto::KernelTagPass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::KernelTagPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
