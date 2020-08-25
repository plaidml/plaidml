// Copyright 2020 Intel Corporation

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
  // StrideInfo outStride;
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
    if (dimCount > 3 || innerSize > maxThreads) {
      return std::numeric_limits<double>::infinity();
    }
    return 1.0 / (innerSize + biggestSize);
  }
};

struct GPUThreadPass : public GPUThreadBase<GPUThreadPass> {
  GPUThreadPass() = default;
  explicit GPUThreadPass(unsigned maxThreads) { this->maxThreads = maxThreads; }
  void threadOp(AffineParallelOp op) {}
  void runOnFunction() final {
    auto func = getFunction();
    // Nest output loops
    for (auto op : func.getOps<AffineParallelOp>()) {
      auto maybeRanges = op.getConstantRanges();
      if (!maybeRanges) {
        continue;
      }
      CostModel model(op, maxThreads);
      auto tileSize =
          findBestTileSize(EvenTilingGenerator(), model, *maybeRanges);
      auto inner = performTiling(op, tileSize);
      op.setAttr("hardware", StringAttr::get("gpu_block", &getContext()));
      inner.setAttr("hardware", StringAttr::get("gpu_thread", &getContext()));
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createGPUThreadPass() {
  return std::make_unique<GPUThreadPass>();
}
std::unique_ptr<mlir::Pass> createGPUThreadPass(unsigned maxThreads) {
  return std::make_unique<GPUThreadPass>(maxThreads);
}

} // namespace pmlc::dialect::pxa.
