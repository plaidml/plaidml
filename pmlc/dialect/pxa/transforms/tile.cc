// Copyright 2020 Intel Corporation

#include <iostream>

#include "mlir/include/mlir/IR/BlockAndValueMapping.h"
#include "pmlc/dialect/pxa/transforms/tile.h"

namespace pmlc::dialect::pxa {

using mlir::AffineParallelOp;

void performTiling(AffineParallelOp op, llvm::ArrayRef<int64_t> tileSizes) {
  mlir::OpBuilder builder(op.getBody(), op.getBody()->begin());
  mlir::Block *outerBody = op.getBody();
  // Verify sizes match
  size_t dimCount = tileSizes.size();
  assert(op.lowerBoundsMap().getNumResults() == dimCount);
  // Fail on no dimensions (TODO: should we handle this case anyway?)
  assert(dimCount > 0);
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
  auto inner = builder.create<AffineParallelOp>(
      op.getLoc(), op.getResultTypes(), lbMap, outerIdxs, ubMap, outerIdxs);
  // Build a map from old indices to new indices
  mlir::BlockAndValueMapping mapper;
  auto innerIdxs = inner.getBody()->getArguments();
  for (unsigned i = 0; i < outerIdxs.size(); ++i) {
    mapper.map(outerIdxs[i], innerIdxs[i]);
  }
  // Clone new instructions into the interior
  auto &innerLoopOps = inner.getBody()->getOperations();
  auto &outerLoopOps = outerBody->getOperations();
  for (auto& origOp : outerLoopOps) {
    if (&origOp == inner.getOperation()) {
      continue;
    }
    Operation* newOp = origOp.clone(mapper);
    innerLoopOps.push_back(newOp);
  }
  outerLoopOps.erase(std::next(outerLoopOps.begin(), 1), outerLoopOps.end());
  // Add a return of the values of the inner to the outer
  builder.setInsertionPointToEnd(op.getBody());
  builder.create<AffineYieldOp>(op.getLoc(), inner.getResults());
  // Update outer step size
  llvm::SmallVector<int64_t, 8> newSteps;
  auto oldSteps = op.steps().cast<ArrayAttr>().getValue();
  for (size_t i = 0; i < dimCount; i++) {
    newSteps.push_back(oldSteps[i].cast<IntegerAttr>().getInt() * tileSizes[i]);
  }
  op.setSteps(newSteps);
}

} // namespace pmlc::dialect::pxa
