// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/tile_accumulate.h"

#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/pxa/transforms/tile.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

AffineParallelOp tileAccumulations(AffineParallelOp op, bool skipTrivial) {
  // Find the originating write and its StrideInfo
  SmallVector<StrideInfo, 4> strides;
  for (auto result : op.getResults()) {
    Optional<StrideInfo> maybeStrideInfo;
    Operation *srcDef = getPrevWriter(op.getResult(0));
    if (auto genericOpInterface =
            dyn_cast_or_null<PxaGenericOpInterface>(srcDef)) {
      PxaMemAccessOperand access = genericOpInterface.getOutputMemAccesses()[0];
      AffineValueMap valueMap = access.getAffineValueMap();
      maybeStrideInfo =
          computeStrideInfo(access.getMemRefType(), valueMap.getAffineMap(),
                            valueMap.getOperands());
    }

    // If we can't fall back to adding a nesting level (to guarentee all
    // accumulations are in the 'inner' loop)
    if (!maybeStrideInfo) {
      auto maybeRanges = op.getConstantRanges();
      assert(maybeRanges &&
             "Cannot tile accumulations on dynamic sized paralllel for");
      if (!skipTrivial) {
        op = performTiling(op, *maybeRanges);
      }
      return op;
    }
    strides.emplace_back(*maybeStrideInfo);
  }

  // Get strides for output
  // Find all the accumulation indexes (stride 0 with respect to output) and
  // tile them into an inner block
  auto ranges = *op.getConstantRanges();
  SmallVector<int64_t, 6> accumTile;
  auto steps = op.getSteps();
  // Track if both inner + outer loops would be used
  bool anyAccum = false;
  bool anyNonAccum = false;
  for (size_t i = 0; i < ranges.size(); i++) {
    auto arg = op.getIVs()[i];
    bool accumArg = false;
    for (auto &si : strides) {
      if (!si.strides.count(arg)) {
        accumArg = true;
        break;
      }
    }
    if (accumArg) {
      // Output stationary, accumulate in inner loop
      anyAccum = true;
      accumTile.push_back(ranges[i]);
    } else {
      // Output non-stationary, outer loop
      anyNonAccum = true;
      accumTile.push_back(steps[i]);
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

struct TileAccumulatePass : public TileAccumulateBase<TileAccumulatePass> {
  void runOnOperation() final {
    auto func = getOperation();
    // Tile only the outermost loops
    func.walk<WalkOrder::PreOrder>([&](AffineParallelOp op) {
      if (!op.getConstantRanges()) {
        signalPassFailure();
        return WalkResult::interrupt();
      }
      tileAccumulations(op, false);
      return WalkResult::skip();
    });
  }
};

} // namespace

std::unique_ptr<Pass> createTileAccumulatePass() {
  return std::make_unique<TileAccumulatePass>();
}

} // namespace pmlc::dialect::pxa.
