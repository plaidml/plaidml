// Copyright 2020, Intel Corporation

#include "pmlc/dialect/pxa/transforms/gpu_thread.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Pass/Pass.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "pmlc/dialect/pxa/transforms/tile.h"

#include "mlir/Support/DebugStringHelper.h"
#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/transforms/autotile.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/tags.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

namespace {
using llvm::DenseMap;
using llvm::DenseSet;
using llvm::SmallVector;

// TODO: More sophisticated cost model
struct CostModel {
  AffineParallelOp op;
  unsigned maxThreads;
  CostModel(AffineParallelOp op, unsigned maxThreads)
      : op(op), maxThreads(maxThreads) {}
  double operator()(ArrayRef<int64_t> tile, double bestCost) const {
    int64_t dimCount = 1;
    int64_t innerSize = 1;
    int64_t biggestSize = 1;
    for (size_t i = 0; i < tile.size(); i++) {
      if (tile[i] > 1) {
        dimCount++;
        innerSize *= tile[i];
        biggestSize = std::max(tile[i], biggestSize);
      }
    }
    // Can't thead over more than 3 dims, can't have more than maxThreads
    if (dimCount > 3 || innerSize > maxThreads) {
      return std::numeric_limits<double>::infinity();
    }
    // We want a lot of threads (innerSize) and preferable at least one big
    // dimension.  This hureristic optimized for that (sort of).
    return 1.0 / (innerSize + biggestSize);
  }
};

struct GPUThreadPass : public GPUThreadBase<GPUThreadPass> {
  GPUThreadPass() = default;
  explicit GPUThreadPass(unsigned maxThreads) { this->maxThreads = maxThreads; }
  void threadOp(AffineParallelOp op) {}
  void runOnFunction() final {
    auto func = getFunction();
    // Nest outermost loops into 'blocks' and 'threads'
    for (auto op : func.getOps<AffineParallelOp>()) {
      gpuThreadParallelOp(maxThreads, op);
    }
  }
};

} // namespace

void gpuThreadParallelOp(unsigned maxThreads, mlir::AffineParallelOp op) {
  auto maybeRanges = op.getConstantRanges();
  if (!maybeRanges) {
    // Fail if we can't compute the ranges at compile time
    return;
  }
  // We want 'logical threads' * 'threads per subgroup' (i.e. subgroupSize)
  // to end up beging about 'maxThreads' threads, and the rest we put into
  // the grid
  unsigned subgroupSize = getIntegerTag(op, subgroupSizeTag(), 1);
  auto goalThreads =
      std::max(1u, static_cast<unsigned>(maxThreads / subgroupSize));
  CostModel model(op, goalThreads);
  auto tileSize = findBestTileSize(EvenTilingGenerator(), model, *maybeRanges);
  auto inner = performTiling(op, tileSize);
  setUnitTag(op, gpuBlockTag());
  setUnitTag(inner, gpuThreadTag());
  setIntegerTag(inner, subgroupSizeTag(), subgroupSize);
}

std::unique_ptr<mlir::Pass> createGPUThreadPass() {
  return std::make_unique<GPUThreadPass>();
}
std::unique_ptr<mlir::Pass> createGPUThreadPass(unsigned maxThreads) {
  return std::make_unique<GPUThreadPass>(maxThreads);
}

} // namespace pmlc::dialect::pxa.
