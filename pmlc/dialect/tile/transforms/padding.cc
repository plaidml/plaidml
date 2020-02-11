// Copyright 2019, Intel Corporation

#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/passes.h"

namespace pmlc::dialect::tile {

using mlir::AffineMap;

struct BoundRange {
  int64_t min;
  int64_t max;
  explicit BoundRange(int64_t val) : min(val), max(val) {}
  BoundRange(int64_t min, int64_t max) : min(min), max(max) {}
  BoundRange operator+(const BoundRange &rhs) const {
    return {min + rhs.min, max + rhs.max};
  }
  BoundRange operator*(const BoundRange &rhs) const {
    return {std::min(std::min(min * rhs.min, max * rhs.min),
                     std::min(min * rhs.max, max * rhs.max)),
            std::max(std::max(min * rhs.min, max * rhs.min),
                     std::max(min * rhs.max, max * rhs.max))};
  }
};

BoundRange merge(const BoundRange &a, const BoundRange &b) {
  return {std::min(a.min, b.min), std::max(a.max, b.max)};
}

typedef std::vector<BoundRange> BoundRanges;

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

BoundRanges computeBounds(AffineMap access, AffineMap lower, AffineMap upper) {
  // We only handle the 'easy' cases.
  assert(access.getNumSymbols() == 0);
  assert(lower.getNumSymbols() == 0);
  assert(upper.getNumSymbols() == 0);
  assert(lower.getNumDims() == 0);
  assert(upper.getNumDims() == 0);
  size_t idxs = access.getNumDims();
  size_t size = access.getNumResults();
  assert(lower.getNumResults() == idxs);
  assert(upper.getNumResults() == idxs);
  assert(access.isPureAffine());
  std::vector<BoundRange> dimRanges;
  for (size_t i = 0; i < idxs; i++) {
    int64_t lowerBoundInt =
        lower.getResult(i).cast<mlir::AffineConstantExpr>().getValue();
    int64_t upperBoundInt =
        upper.getResult(i).cast<mlir::AffineConstantExpr>().getValue();
    dimRanges.emplace_back(lowerBoundInt, upperBoundInt - 1);
  }
  ComputeRangeVisitor visitor(std::move(dimRanges));
  std::vector<BoundRange> outRanges;
  for (size_t i = 0; i < size; i++) {
    outRanges.push_back(visitor.visit(access.getResult(i)));
  }
  return dimRanges;
}

/*
struct BoundsInfo {
  std::map<AggregationKind, TensorBounds> bounds;
};
*/

void PaddingPass::runOnFunction() {
  auto func = getFunction();
  func.walk([&](ContractionOp op) {
    if (op.getNumSymbols()) {
      op.emitError("padding cannot be run on symbolic contractions");
      return mlir::WalkResult::interrupt();
    }
    if (!op.lowerBounds().hasValue() || !op.upperBounds().hasValue()) {
      op.emitError("contraction bounds must be computed");
      return mlir::WalkResult::interrupt();
    }
    for (size_t i = 0; i < op.getNumTensors(); i++) {
      auto ranges = computeBounds(op.getSourceMap(i), *op.lowerBounds(),
                                  *op.upperBounds());
    }
    return mlir::WalkResult::advance();
  });
}

std::unique_ptr<mlir::Pass> createPaddingPass() {
  return std::make_unique<PaddingPass>();
}

} // namespace pmlc::dialect::tile
