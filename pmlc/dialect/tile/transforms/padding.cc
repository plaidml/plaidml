// Copyright 2020 Intel Corporation

#include "pmlc/dialect/tile/transforms/padding.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"

namespace pmlc::dialect::tile {

using mlir::AffineExpr;
using mlir::AffineMap;

namespace {

class ComputeRangeVisitor
    : public mlir::AffineExprVisitor<ComputeRangeVisitor, BoundRange> {
public:
  explicit ComputeRangeVisitor(std::vector<BoundRange> dimRanges)
      : dimRanges(std::move(dimRanges)) {}

  BoundRange visitMulExpr(mlir::AffineBinaryOpExpr expr) {
    return visit(expr.getLHS()) * visit(expr.getRHS());
  }

  BoundRange visitAddExpr(mlir::AffineBinaryOpExpr expr) {
    return visit(expr.getLHS()) + visit(expr.getRHS());
  }

  BoundRange visitDimExpr(mlir::AffineDimExpr expr) {
    return dimRanges[expr.getPosition()];
  }

  BoundRange visitSymbolExpr(mlir::AffineSymbolExpr expr) {
    llvm_unreachable("Invalid symbol in ComputeRangeVisitor");
  }

  BoundRange visitConstantExpr(mlir::AffineConstantExpr expr) {
    return BoundRange(expr.getValue());
  }

  BoundRange visitModExpr(mlir::AffineBinaryOpExpr expr) {
    llvm_unreachable("Invalid mod in ComputeRangeVisitor");
  }

  BoundRange visitCeilDivExpr(mlir::AffineBinaryOpExpr expr) {
    llvm_unreachable("Invalid ceil in ComputeRangeVisitor");
  }

  BoundRange visitFloorDivExpr(mlir::AffineBinaryOpExpr expr) {
    llvm_unreachable("Invalid floor in ComputeRangeVisitor");
  }

private:
  std::vector<BoundRange> dimRanges;
};

} // namespace

BoundRange BoundRange::operator+(const BoundRange &rhs) const {
  return {min + rhs.min, max + rhs.max};
}

BoundRange BoundRange::operator*(const BoundRange &rhs) const {
  return {std::min(std::min(min * rhs.min, max * rhs.min),
                   std::min(min * rhs.max, max * rhs.max)),
          std::max(std::max(min * rhs.min, max * rhs.min),
                   std::max(min * rhs.max, max * rhs.max))};
}

llvm::SmallVector<BoundRange, 4>
computePaddingBounds(AffineMap access, AffineMap lower, AffineMap upper) {
  // We only handle the 'easy' cases.
  assert(access.getNumSymbols() == 0);
  assert(lower.getNumSymbols() == 0);
  assert(upper.getNumSymbols() == 0);
  assert(lower.getNumDims() == 0);
  assert(upper.getNumDims() == 0);
  unsigned idxs = access.getNumDims();
  unsigned size = access.getNumResults();
  assert(lower.getNumResults() == idxs);
  assert(upper.getNumResults() == idxs);
  std::vector<BoundRange> dimRanges;
  for (unsigned i = 0; i < idxs; i++) {
    int64_t lowerBoundInt =
        lower.getResult(i).cast<mlir::AffineConstantExpr>().getValue();
    int64_t upperBoundInt =
        upper.getResult(i).cast<mlir::AffineConstantExpr>().getValue();
    dimRanges.emplace_back(lowerBoundInt, upperBoundInt);
  }
  ComputeRangeVisitor visitor(std::move(dimRanges));
  llvm::SmallVector<BoundRange, 4> outRanges;
  for (unsigned i = 0; i < size; i++) {
    outRanges.push_back(visitor.visit(access.getResult(i)));
  }
  return outRanges;
}

} // namespace pmlc::dialect::tile
