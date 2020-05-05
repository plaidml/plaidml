// Copyright 2020 Intel Corporation

#pragma once

#include <functional>
#include <memory>

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::dialect::pxa {

std::unique_ptr<mlir::Pass> createAutoTileExamplePass();

struct StencilCost {
  double throughput;
  unsigned startupCost;
};

using StencilCostFunction =
    std::function<StencilCost(llvm::ArrayRef<unsigned>)>;

std::unique_ptr<mlir::Pass> createTestStrideInfoPass();

std::unique_ptr<mlir::Pass> createXSMMStencilPass(unsigned numThreads,
                                                  StencilCostFunction costFn);

} // namespace pmlc::dialect::pxa
