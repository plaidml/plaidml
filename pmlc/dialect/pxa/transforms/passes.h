// Copyright 2020 Intel Corporation

#pragma once

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

struct StencilPassOptions
    : public mlir::PassPipelineOptions<StencilPassOptions> {
  Option<int> numberOfThreadsOption{
      *this, "affine-stencil-number-threads",
      llvm::cl::desc("Specifies number of threads for the stencilling pass")};
};

struct StencilPass : public mlir::FunctionPass<StencilPass> {
  StencilPass() = default;
  StencilPass(const StencilPass &) {}
  explicit StencilPass(const StencilPassOptions &options);

  Option<int> numberOfThreadsOption{
      *this, "affine-stencil-number-threads",
      llvm::cl::desc("Specifies number of threads for the stencilling pass")};
  void runOnFunction() final;
};

StencilPass::StencilPass(const StencilPassOptions &options) {
  numberOfThreadsOption.setValue(options.numberOfThreadsOption);
}

void createStencilPass(mlir::OpPassManager &pm, const StencilPassOptions &);

} // namespace pmlc::dialect::pxa
