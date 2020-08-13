// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/tile.h"

namespace pmlc::dialect::pxa {

using mlir::AffineParallelOp;

AffineParallelOp performTiling(AffineParallelOp op,
                               llvm::ArrayRef<int64_t> tileSizes) {
  // Extract steps (TODO: this should be a utility on affine.parallel)
  auto oldStepsArray = op.steps().cast<ArrayAttr>().getValue();
  llvm::SmallVector<int64_t, 6> oldSteps;
  for (auto ia : oldStepsArray) {
    oldSteps.push_back(ia.cast<IntegerAttr>().getInt());
  }
  // Make builder
  mlir::OpBuilder builder(op.getBody(), op.getBody()->begin());
  mlir::Block *outerBody = op.getBody();
  // Verify sizes match
  size_t dimCount = tileSizes.size();
  assert(op.lowerBoundsMap().getNumResults() == dimCount);
  // Fail on no dimensions (TODO: should we handle this case anyway?)
  assert(dimCount > 0);
  // Check that tile sizes is a multiple of original steps
  for (size_t i = 0; i < dimCount; i++) {
    assert(tileSizes[i] % oldSteps[i] == 0);
  }
  // Make the maps for the inner parallel
  llvm::SmallVector<mlir::AffineExpr, 8> lbExprs;
  llvm::SmallVector<mlir::AffineExpr, 8> ubExprs;
  for (size_t i = 0; i < dimCount; i++) {
    auto outerDim = builder.getAffineDimExpr(i);
    auto tileSize = builder.getAffineConstantExpr(tileSizes[i]);
    lbExprs.push_back(outerDim);
    ubExprs.push_back(outerDim + tileSize);
  }
  auto lbMap = AffineMap::get(dimCount, 0, lbExprs, op.getContext());
  auto ubMap = AffineMap::get(dimCount, 0, ubExprs, op.getContext());
  auto outerIdxs = outerBody->getArguments();
  // Make the inner parallel for (abve all other code);
  llvm::SmallVector<mlir::AtomicRMWKind, 8> reductions;
  for (Attribute attr : op.reductions()) {
    auto intAttr = attr.dyn_cast<IntegerAttr>();
    reductions.push_back(*mlir::symbolizeAtomicRMWKind(intAttr.getInt()));
  }
  auto inner = builder.create<AffineParallelOp>(
      op.getLoc(), op.getResultTypes(), reductions, lbMap, outerIdxs, ubMap,
      outerIdxs);
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
  llvm::SmallVector<int64_t, 8> newSteps;
  inner.setSteps(oldSteps);
  op.setSteps(tileSizes);
  return inner;
}

} // namespace pmlc::dialect::pxa
