// Copyright 2020 Intel Corporation

#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "pmlc/dialect/pxa/transforms/autotile.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/pxa/transforms/tile.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/util/tags.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

namespace {

static constexpr llvm::StringLiteral kCpuThreadTag = "cpuThread";

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

  void runOnFunction() final {
    auto func = getFunction();
    CostModel model(threads);
    // Nest outermost loops into 'blocks' and 'threads'
    func.walk<WalkOrder::PreOrder>([&](AffineParallelOp op) {
      processOp(op, model);
      return WalkResult::skip();
    });
  }

  void processOp(AffineParallelOp op, CostModel model) {
    auto maybeRanges = op.getConstantRanges();
    if (!maybeRanges) {
      // Fail if we can't compute the ranges at compile time
      return;
    }
    auto tileSize =
        findBestTileSize(EvenTilingGenerator(), model, *maybeRanges);
    // Invert tiling (we want 'threads' on the outer loop
    for (size_t i = 0; i < tileSize.size(); i++) {
      tileSize[i] = (*maybeRanges)[i] / tileSize[i];
    }
    // Tile and tag
    performTiling(op, tileSize);
    setUnitTag(op, kCpuThreadTag);
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
