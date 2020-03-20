// Copyright 2020 Intel Corporation

#pragma once

#include <memory>
#include <utility>

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
  StencilPass() : costerSet(false) {}
  StencilPass(const StencilPass &) : costerSet(false) {}

  Option<unsigned> numThreads{
      *this, "threads",
      llvm::cl::desc("Specifies number of threads for the stencilling pass")};
  void runOnFunction() final;
  void setCoster(std::function<std::pair<double, unsigned>(const unsigned *,
                                                           const unsigned)>
                     costerIn) {
    coster = costerIn;
    costerSet = true;
  }

  bool costerSet;
  std::function<std::pair<double, unsigned>(const unsigned *, const unsigned)>
      coster;
};

std::unique_ptr<mlir::Pass> createStencilPass();
std::unique_ptr<mlir::Pass> createStencilPassWithCoster(
    std::function<std::pair<double, unsigned>(const unsigned *, const unsigned)>
        coster);

} // namespace pmlc::dialect::pxa
