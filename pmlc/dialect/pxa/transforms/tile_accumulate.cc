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
  // Track if both inner + outer loops would bee used
  bool anyAccum = false;
  bool anyNonAccum = false;
  for (size_t i = 0; i < ranges.size(); i++) {
    auto arg = op.getIVs()[i];
    if (si.strides.count(arg)) {
      // Output non-stationary, outer loop
      anyNonAccum = true;
      accumTile.push_back(steps[i].cast<IntegerAttr>().getInt());
    } else {
      // Output stationary, accumulate in inner loop
      anyAccum = true;
      accumTile.push_back(ranges[i]);
    }
  }
  // Check if both loops were used
  bool nonTrivial = anyAccum && anyNonAccum;
  // Tile if needed or if we always want fixed depth
  if (nonTrivial || !skipTrivial) {
    op = performTiling(op, accumTile);
  }
  return op;
}

namespace {
using llvm::DenseMap;
using llvm::DenseSet;
using llvm::SmallVector;
using mlir::BlockArgument;

struct TileAccumulatePass : public TileAccumulateBase<TileAccumulatePass> {
  void runOnFunction() final {
    auto func = getFunction();
    // Tile only the outermost loops
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
