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

class FixStridesPass final : public CompilePass {
 public:
  explicit FixStridesPass(const proto::FixStridesPass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::FixStridesPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
