// Copyright 2020 Intel Corporation

#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
class FuncOp;
class Pass;
template <typename T>
class OpPassBase;
} // namespace mlir

namespace pmlc::dialect::pxa {

struct StencilPass : public mlir::FunctionPass<StencilPass> {
  StencilPass() = default;
  StencilPass(const StencilPass &) {}

  Option<unsigned> numThreads{
      *this, "threads",
      llvm::cl::desc("Specifies number of threads for the stencilling pass")};
  void runOnFunction() final;
};

std::unique_ptr<mlir::Pass> createStencilPass();

} // namespace pmlc::dialect::pxa
