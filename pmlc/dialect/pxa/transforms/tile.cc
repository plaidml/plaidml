// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/tile.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

AffineParallelOp performTiling(AffineParallelOp op,
                               ArrayRef<int64_t> tileSizes) {
  // Make builder
  OpBuilder builder(op.getBody(), op.getBody()->begin());
  Block *outerBody = op.getBody();
  // Verify sizes match
  size_t rank = tileSizes.size();
  assert(op.lowerBoundsMap().getNumResults() == rank);
  // Check that tile sizes is a multiple of original steps
  auto steps = op.getSteps();
  for (size_t i = 0; i < rank; i++) {
    assert(tileSizes[i] % steps[i] == 0);
  }
  // Make the maps for the inner parallel
  SmallVector<AffineMap, 8> lbMaps;
  SmallVector<AffineMap, 8> ubMaps;
  for (size_t i = 0; i < rank; i++) {
    AffineExpr outerExpr = builder.getAffineDimExpr(i);
    AffineExpr tileExpr = builder.getAffineConstantExpr(tileSizes[i]);
    lbMaps.push_back(AffineMap::get(rank, 0, outerExpr));
    ubMaps.push_back(AffineMap::get(rank, 0, outerExpr + tileExpr));
  }
  auto outerIdxs = outerBody->getArguments();
  // Make the inner parallel for (above all other code);
  SmallVector<arith::AtomicRMWKind, 8> reductions;
  for (Attribute attr : op.reductions()) {
    auto intAttr = attr.dyn_cast<IntegerAttr>();
    reductions.push_back(*arith::symbolizeAtomicRMWKind(intAttr.getInt()));
  }
  auto inner = builder.create<AffineParallelOp>(
      op.getLoc(), op.getResultTypes(), reductions, lbMaps, outerIdxs, ubMaps,
      outerIdxs, steps);
  // Splice instructions into the interior
  auto &innerLoopOps = inner.getBody()->getOperations();
  auto &outerLoopOps = outerBody->getOperations();
  innerLoopOps.splice(std::prev(innerLoopOps.end()), outerLoopOps,
                      std::next(outerLoopOps.begin(), 1), outerLoopOps.end());
  // Replace old indices with new indices
  auto innerIdxs = inner.getBody()->getArguments();
  for (unsigned i = 0; i < outerIdxs.size(); ++i) {
    outerIdxs[i].replaceAllUsesWith(innerIdxs[i]);
  }
  unsigned numIdxs = inner.lowerBoundsMap().getNumInputs();
  for (unsigned i = 0; i < numIdxs; ++i) {
    inner.setOperand(i, outerIdxs[i]);
    inner.setOperand(i + numIdxs, outerIdxs[i]);
  }
  // Add a return of the values of the inner to the outer
  builder.setInsertionPointToEnd(op.getBody());
  builder.create<AffineYieldOp>(op.getLoc(), inner.getResults());
  // Update outer step size
  op.setSteps(tileSizes);
  return inner;
}

AffineParallelOp undoTiling(AffineParallelOp op, ArrayRef<int64_t> tileSizes) {
  // Make builder
  OpBuilder builder(op.getBody(), op.getBody()->begin());
  Block *outerBody = op.getBody();
  // Verify sizes match
  size_t dimCount = tileSizes.size();
  assert(op.lowerBoundsMap().getNumResults() == dimCount);
  // Check that we can undo steps
  auto steps = op.getSteps();
  for (size_t i = 0; i < dimCount; i++) {
    assert(steps[i] % tileSizes[i] == 0);
    steps[i] /= tileSizes[i];
  }

  // Check if first operation is AffineParallelOp (inner loop after tiling)
  Operation &inner = outerBody->front();
  auto innerOp = dyn_cast<AffineParallelOp>(&inner);
  assert(innerOp);

  // Check if last operation is AffineYieldOp
  Operation &yield = outerBody->back();
  auto yieldOp = dyn_cast<AffineYieldOp>(&yield);
  assert(yieldOp);

  // Finished with checks, first remove the redundant AffineYieldOp
  yieldOp.erase();

  // Replace old indices with new indices
  auto &outerLoopOps = outerBody->getOperations();
  auto &innerLoopOps = innerOp.getBody()->getOperations();

  auto outerIdxs = outerBody->getArguments();
  auto innerIdxs = innerOp.getBody()->getArguments();
  for (unsigned i = 0; i < innerIdxs.size(); ++i) {
    innerIdxs[i].replaceAllUsesWith(outerIdxs[i]);
  }

  // Move the ops from inner back to outer
  outerLoopOps.splice(std::prev(outerLoopOps.end()), innerLoopOps,
                      innerLoopOps.begin(), innerLoopOps.end());

  // Remove empty inner loop
  innerOp.erase();

  // Set orginal steps
  op.setSteps(steps);

  return op;
}

} // namespace pmlc::dialect::pxa
