// Copyright 2019, Intel Corporation

#include <iostream>

#include "pmlc/dialect/pxa/transforms/tile.h"

namespace pmlc::dialect::pxa {

void Tile(AffineParallelOp op, llvm::ArrayRef<int64_t> tileSizes) {
  auto builder = op.getBodyBuilder();
  mlir::Block* outerBody = op.getBody();
  // Verify sizes match
  size_t dimCount = tileSizes.size();
  assert(op.lowerBoundsMap().numResults() == dimCount());
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
  auto lbMap = AffineMap::get(dimCount, 0, lbExprs);
  auto ubMap = AffineMap::get(dimCount, 0, ubExprs);
  // TODO: Maybe fix ValueRange?
  llvm::SmallVector<mlir::Value, 8> outerIdxs;
  for (size_t i = 0; i < outerBody->getNumArguments(); i++) {
    outerIdxs.push_back(outerBody->getArgument(i));
  }
  // Make the inner parallel for
  auto inner = builder.create<AffineParallelOp>(op.getLoc(), lbMap, outerIdxs, ubMap, outerIdxs);
  // Splice instructions into the interior
  auto& innerLoopOps = inner.getBody()->getOperations();
  auto& outerLoopOps = outerBody->getOperations();
  innerLoopOps.splice(std::prev(innerLoopOps.end()), outerLoopOps, outerLoopOps.begin(),
                      std::prev(outerLoopOps.end(), 2));
  // Update outer step size
  llvm::SmallVector<int64_t, 8> newSteps;
  auto oldSteps = op.steps().cast<ArrayAttr>().getValue();
  for (size_t i = 0; i < dimCount; i++) {
    newSteps.push_back(oldSteps[i].cast<IntegerAttr>().getInt() * tileSizes[i]);
  }
  op.setAttr("steps", builder.getI64ArrayAttr(newSteps));
}

}  // namespace pmlc::dialect::pxa
