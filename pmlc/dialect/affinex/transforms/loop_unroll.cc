// Copyright 2020 Intel Corporation

#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Transforms/LoopUtils.h"
#include "pmlc/dialect/affinex/transforms/pass_detail.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::affinex {

struct AffinexLoopUnroll : public AffinexLoopUnrollBase<AffinexLoopUnroll> {
  void runOnFunction() override {
    DenseMap<Operation *, uint64_t> opCount;

    getFunction().walk([&](Operation *op) {
      auto count = opCount[op];
      if (isa<AffineForOp>(op)) {
        AffineForOp forOp = dyn_cast<AffineForOp>(op);
        Optional<uint64_t> tripCount = getConstantTripCount(forOp);
        if (tripCount.hasValue()) {
          count *= *tripCount;
        } else {
          count = operationLimit.getValue() + 1;
        }
      } else if (isa<AffineYieldOp>(op)) {
        // don't count these op(s)
        count = 0;
      } else if (isa<AffineParallelOp>(op)) {
        // stop unrolling on these op(s)
        count = operationLimit.getValue() + 1;
      } else {
        // otherwise, count the op
        count++;
      }
      opCount[op->getParentOp()] += count;

      if (isa<AffineForOp>(op) && count <= operationLimit.getValue()) {
        AffineForOp forOp = dyn_cast<AffineForOp>(op);
        loopUnrollFull(forOp);
      }
    });
  }
};

std::unique_ptr<Pass> createAffinexLoopUnroll() {
  return std::make_unique<AffinexLoopUnroll>();
}
} // namespace pmlc::dialect::affinex
