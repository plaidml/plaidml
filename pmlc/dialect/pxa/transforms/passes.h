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

std::unique_ptr<mlir::Pass> createVectorizeExamplePass();

std::unique_ptr<mlir::Pass> createFusionPass();

std::unique_ptr<mlir::Pass> createLocalizePass();

std::unique_ptr<mlir::Pass> createMemRefDataFlowOptPass();

std::unique_ptr<mlir::Pass> createResizeTmpsPass();

struct StencilCost {
  double throughput;
  unsigned startupCost;
};

using StencilCostFunction = std::function<StencilCost(llvm::ArrayRef<int64_t>)>;

std::unique_ptr<mlir::Pass> createTestStrideInfoPass();

std::unique_ptr<mlir::Pass> createTestIndirectUsesIteratorPass();

std::unique_ptr<mlir::Pass> createTestIndirectValuesIteratorPass();

std::unique_ptr<mlir::Pass> createStencilGEMMPass(unsigned numThreads,
                                                  StencilCostFunction costFn);

} // namespace pmlc::dialect::pxa
