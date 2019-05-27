// Copyright 2018, Intel Corporation

#pragma once

#include <map>
#include <string>

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

class ConstantPropagatePass final : public CompilePass {
 public:
  explicit ConstantPropagatePass(const proto::ConstantPropagatePass& options) {}
  void Apply(CompilerState* state) const final;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
