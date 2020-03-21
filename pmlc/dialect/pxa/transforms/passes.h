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

struct StencilCost {
  double throughput;
  unsigned startupCost;
};

using StencilCostFunction =
    std::function<StencilCost(llvm::ArrayRef<unsigned>)>;

std::unique_ptr<mlir::Pass> createStencilPass(unsigned numThreads,
                                              StencilCostFunction costFn);

} // namespace pmlc::dialect::pxa
