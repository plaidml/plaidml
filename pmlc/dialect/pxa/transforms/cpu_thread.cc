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
#include "pmlc/util/tags.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

namespace {
using llvm::DenseMap;
using llvm::DenseSet;
using llvm::SmallVector;

// Pick the tiling that is as large as possible without going over maxThreads
struct CostModel {
  unsigned maxThreads;
  explicit CostModel(unsigned maxThreads) : maxThreads(maxThreads) {}
  double operator()(ArrayRef<int64_t> tile, double bestCost) const {
    int64_t innerSize = 1;
    for (size_t i = 0; i < tile.size(); i++) {
      innerSize *= tile[i];
    }
    if (innerSize > maxThreads) {
      return std::numeric_limits<double>::infinity();
    }
    return 1.0 / innerSize;
  }
};

struct CPUThreadPass : public CPUThreadBase<CPUThreadPass> {
  CPUThreadPass() = default;
  explicit CPUThreadPass(unsigned threads) { this->threads = threads; }
  void threadOp(AffineParallelOp op) {}
  void runOnFunction() final {
    auto func = getFunction();
    CostModel model(threads);
    // Nest outermost loops into 'blocks' and 'threads'
    for (auto op : func.getOps<AffineParallelOp>()) {
      auto maybeRanges = op.getConstantRanges();
      if (!maybeRanges) {
        // Fail if we can't compute the ranges at compile time
        continue;
      }
      auto tileSize =
          findBestTileSize(EvenTilingGenerator(), model, *maybeRanges);
      // Invert tiling (we want 'threads' on the outer loop
      for (size_t i = 0; i < tileSize.size(); i++) {
        tileSize[i] = (*maybeRanges)[i] / tileSize[i];
      }
      // Tile and tag
      performTiling(op, tileSize);
      setUnitTag(op, cpuThreadTag());
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createCPUThreadPass() {
  return std::make_unique<CPUThreadPass>();
}
std::unique_ptr<mlir::Pass> createCPUThreadPass(unsigned threads) {
  return std::make_unique<CPUThreadPass>(threads);
}

} // namespace pmlc::dialect::pxa.
