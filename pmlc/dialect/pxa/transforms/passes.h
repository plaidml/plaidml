// Copyright 2020 Intel Corporation

#pragma once

#include <functional>
#include <memory>

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"

namespace pmlc::dialect::pxa {

struct StencilCost {
  double throughput;
  unsigned startupCost;
};

using StencilCostFunction = std::function<StencilCost(llvm::ArrayRef<int64_t>)>;

std::unique_ptr<mlir::Pass> createAffineNormalizePass();

std::unique_ptr<mlir::Pass> createAutoTileExamplePass();

std::unique_ptr<mlir::Pass> createBufferPlacementPass();

std::unique_ptr<mlir::Pass>
createFusionPass(int64_t memoryActivityThreshold = 0);

std::unique_ptr<mlir::Pass> createLocalizePass();

std::unique_ptr<mlir::Pass> createMemRefDataFlowOptPass();

std::unique_ptr<mlir::Pass> createNestLoopsPass();

std::unique_ptr<mlir::Pass> createNestLoopsPass(unsigned minLoopIVs);

std::unique_ptr<mlir::Pass> createResizeTmpsPass();

std::unique_ptr<mlir::Pass> createStencilGEMMPass(unsigned numThreads,
                                                  StencilCostFunction costFn);

std::unique_ptr<mlir::Pass> createSubgroupsPass();

std::unique_ptr<mlir::Pass> createTestStrideInfoPass();

std::unique_ptr<mlir::Pass> createTestIndirectUsesIteratorPass();

std::unique_ptr<mlir::Pass> createTestIndirectValuesIteratorPass();

std::unique_ptr<mlir::Pass> createTileAccumulatePass();

std::unique_ptr<mlir::Pass> createVectorizeExamplePass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/dialect/pxa/transforms/passes.h.inc"

} // namespace pmlc::dialect::pxa
