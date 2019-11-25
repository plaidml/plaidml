// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "tile/codegen/codegen.pb.h"

namespace pmlc {
namespace dialect {
namespace stripe {

struct AutoStencilPass : public mlir::FunctionPass<AutoStencilPass> {
  explicit AutoStencilPass(const vertexai::tile::codegen::proto::MLIR_AutoStencilPass& options) : options(options) {}
  void runOnFunction() override;

  vertexai::tile::codegen::proto::MLIR_AutoStencilPass options;
};

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
