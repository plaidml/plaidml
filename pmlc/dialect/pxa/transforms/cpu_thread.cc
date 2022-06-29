// Copyright 2020 Intel Corporation

#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
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
  ArrayRef<int64_t> strides;

  explicit CostModel(unsigned maxThreads, ArrayRef<int64_t> strides)
      : maxThreads(maxThreads), strides(strides) {}

  double operator()(ArrayRef<int64_t> tile, double bestCost) const {
    int64_t innerSize = 1;
    int64_t maxStride = 1;
    for (size_t i = 0; i < tile.size(); i++) {
      innerSize *= tile[i];
      if (tile[i] != 1)
        maxStride = std::max(maxStride, strides[i]);
    }
    if (innerSize > maxThreads) {
      return std::numeric_limits<double>::infinity();
    }
    return (1.0 / innerSize) + (1.0 / maxStride);
  }
};

struct CPUThreadPass : public CPUThreadBase<CPUThreadPass> {
  CPUThreadPass() = default;
  explicit CPUThreadPass(unsigned threads) { this->threads = threads; }

  void runOnOperation() final {
    auto func = getOperation();
    // Nest outermost loops into 'blocks' and 'threads'
    func.walk<WalkOrder::PreOrder>([&](AffineParallelOp op) {
      for (auto bodyItr = op.getBody()->begin(); bodyItr != op.getBody()->end();
           bodyItr++) {
        if (isa<mlir::memref::AllocOp>(bodyItr)) {
          auto allocOp = cast<mlir::memref::AllocOp>(bodyItr);
          OpBuilder builder(allocOp);
          auto memRefType = MemRefType::get(
              allocOp.getType().getShape(), allocOp.getType().getElementType(),
              MemRefLayoutAttrInterface(), allocOp.getType().getMemorySpace());
          auto allocaOp = builder.create<mlir::memref::AllocaOp>(
              op.getBody()->getParentOp()->getParentOp()->getLoc(), memRefType);
          allocOp.replaceAllUsesWith(allocaOp.getResult());
          allocOp.erase();
        }
      }

      processOp(op);
      return WalkResult::skip();
    });
  }

  void processOp(AffineParallelOp op) {
    auto maybeRanges = op.getConstantRanges();
    if (!maybeRanges) {
      // Fail if we can't compute the ranges at compile time
      return;
    }

    /*  SmallVector<int64_t> strides(op.getNumDims(), 0);
      if (auto lastWriter =
              dyn_cast_or_null<PxaReduceOp>(getPrevWriter(op.getResult(0)))) {
        if (Optional<StrideInfo> si = computeStrideInfo(lastWriter)) {
          for (BlockArgument arg : op.getIVs()) {
            strides[arg.getArgNumber()] = si->strides[arg];
          }
        }
      }

      CostModel model(threads, strides);
      auto tileSize =
          findBestTileSize(EvenTilingGenerator(), model, *maybeRanges);
      // Invert tiling (we want 'threads' on the outer loop
      for (size_t i = 0; i < tileSize.size(); i++) {
        tileSize[i] = (*maybeRanges)[i] / tileSize[i];
      }
  */
    SmallVector<int64_t, 8> tileSize;
    for (int i = 0; i < op.getNumDims(); i++) {
      if ((*maybeRanges)[i] == threads) {
        tileSize.push_back(1);
      } else {
        tileSize.push_back((*maybeRanges)[i]);
      }
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
