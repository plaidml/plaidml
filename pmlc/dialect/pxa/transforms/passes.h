// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
class FuncOp;
class Pass;
template <typename T>
class OpPassBase;
} // namespace mlir

namespace pmlc::dialect::pxa {

struct StencilPass : public mlir::FunctionPass<StencilPass> {
  void runOnFunction() final;
};

std::unique_ptr<mlir::Pass> createStencilPass();

} // namespace pmlc::dialect::pxa
