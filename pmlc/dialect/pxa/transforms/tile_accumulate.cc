// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/tile_accumulate.h"

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
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

namespace {
using llvm::DenseMap;
using llvm::DenseSet;
using llvm::SmallVector;
using mlir::BlockArgument;

bool isAccumulation(AffineParallelOp op) {
  bool tag = false;
  auto argRange = op.getIVs();
  auto parallelLoopNum = argRange.size() < 2 ? argRange.size() : 2;
  op.walk([&](PxaReduceOp reduce) {
    auto range = reduce.idxs();
    for (size_t i = 0; i < parallelLoopNum; i++) {
      auto firstArg = std::find(range.begin(), range.end(), argRange[i]);
      tag = firstArg == range.end() || tag;
    }
  });
  IVLOG(1, "the accumutation tag is " << tag);
  return tag;
}

AffineParallelOp tileAccumulations(AffineParallelOp op, bool skipTrivial) {
  // Find the originating reduce
  assert(op.getNumResults() == 1);
  auto srcDef = getOriginalDef(op.getResult(0));
  auto red = mlir::cast<PxaReduceOp>(srcDef);
  // Get strides for output
  auto si = *computeStrideInfo(red);
  // Find all the accumulation indexes (stride 0 with respect to output) and
  // tile them into an inner block
  auto ranges = *op.getConstantRanges();
  SmallVector<int64_t, 6> accumTile;
  auto steps = op.steps().cast<ArrayAttr>().getValue();
  bool anyAccum = false;
  bool anyNonAccum = false;
  for (size_t i = 0; i < ranges.size(); i++) {
    auto arg = op.getIVs()[i];
    if (si.strides.count(arg)) {
      anyNonAccum = true;
      accumTile.push_back(steps[i].cast<IntegerAttr>().getInt());
    } else {
      anyAccum = true;
      accumTile.push_back(ranges[i]);
    }
  }
  bool nonTrivial = anyAccum && anyNonAccum;
  if (nonTrival || !skipTrivial) {
    op = performTiling(op, accumTile);
  }
  return op;
}

struct TileAccumulatePass : public TileAccumulateBase<TileAccumulatePass> {
  void runOnFunction() final {
    auto func = getFunction();
    // TIle only the outermost loops
    for (auto op : func.getBody().getOps<AffineParallelOp>()) {
      if (op.getNumResults() == 1) {
        tileAccumulations(op, true);
      }
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createTileAccumulatePass() {
  return std::make_unique<TileAccumulatePass>();
}

} // namespace pmlc::dialect::pxa.
