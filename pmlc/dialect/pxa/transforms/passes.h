// Copyright 2020 Intel Corporation

#pragma once

#include <functional>
#include <memory>

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class Pass;
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
