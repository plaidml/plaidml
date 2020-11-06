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

using StencilCostFunction = std::function<StencilCost(
    llvm::ArrayRef<int64_t>, llvm::ArrayRef<mlir::Type>)>;

std::unique_ptr<mlir::Pass> createAffineNormalizePass();
std::unique_ptr<mlir::Pass> createAffineNormalizePass(bool promote);

std::unique_ptr<mlir::Pass> createAutoTileExamplePass();

std::unique_ptr<mlir::Pass> createBufferPlacementPass();

std::unique_ptr<mlir::Pass> createCachePass();

std::unique_ptr<mlir::Pass> createCPUThreadPass();
std::unique_ptr<mlir::Pass> createCPUThreadPass(unsigned threads);

std::unique_ptr<mlir::Pass>
createFusionPass(int64_t memoryActivityThreshold = 0, bool exactlyMatch = false,
                 bool tiledFusion = false, int64_t loopDepth = 0,
                 bool singleOutput = false);

std::unique_ptr<mlir::Pass> createGPUThreadPass();
std::unique_ptr<mlir::Pass> createGPUThreadPass(unsigned maxThreads);

std::unique_ptr<mlir::Pass> createLocalizePass();

std::unique_ptr<mlir::Pass>
createMemRefDataFlowOptPass(bool onlyParallelNested = false);

std::unique_ptr<mlir::Pass> createNestLoopsPass();
std::unique_ptr<mlir::Pass> createNestLoopsPass(unsigned minLoopIVs);

std::unique_ptr<mlir::Pass>
createResizeTmpsPass(bool onlyParallelNested = false);

std::unique_ptr<mlir::Pass> createStencilGEMMPass();
std::unique_ptr<mlir::Pass> createStencilGEMMPass(unsigned numThreads,
                                                  bool doBatch,
                                                  StencilCostFunction costFn);

std::unique_ptr<mlir::Pass> createSubgroupsPass();

std::unique_ptr<mlir::Pass> createTestStrideInfoPass();

std::unique_ptr<mlir::Pass> createTestIndirectUsesIteratorPass();

std::unique_ptr<mlir::Pass> createTestIndirectValuesIteratorPass();

std::unique_ptr<mlir::Pass> createTileAccumulatePass();

std::unique_ptr<mlir::Pass> createVectorizePass();

std::unique_ptr<mlir::Pass> createVectorizePass(mlir::StringRef strategy,
                                                unsigned vectorWidth = 8);

std::unique_ptr<mlir::Pass> createSimplifyWithConstraintsPass();

std::unique_ptr<mlir::Pass> createReorderLayoutsPass();
std::unique_ptr<mlir::Pass> createReorderLayoutsPass(bool allowReorder);

std::unique_ptr<mlir::Pass> createVectorizeMemPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/dialect/pxa/transforms/passes.h.inc"

} // namespace pmlc::dialect::pxa
