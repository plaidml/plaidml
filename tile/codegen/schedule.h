// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

class SchedulePass final : public CompilePass {
 public:
  explicit SchedulePass(const proto::SchedulePass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::SchedulePass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
