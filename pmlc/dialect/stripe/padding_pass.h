// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "tile/codegen/codegen.pb.h"

namespace pmlc {
namespace dialect {
namespace stripe {

struct PaddingPass : public mlir::FunctionPass<PaddingPass> {
  explicit PaddingPass(const vertexai::tile::codegen::proto::StripeMLIRPadPass& options) : options(options) {}
  void runOnFunction() override;

  vertexai::tile::codegen::proto::StripeMLIRPadPass options;
};

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
