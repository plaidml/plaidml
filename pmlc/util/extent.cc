// Copyright 2021, Intel Corporation

#include "mlir/IR/AffineExprVisitor.h"

#include "pmlc/util/extent.h"

using namespace mlir; // NOLINT

namespace pmlc::util {

class ExtentComputer : public AffineExprVisitor<ExtentComputer, Extent> {
public:
  explicit ExtentComputer(ArrayRef<Extent> extents) : vars(extents) {}

  Extent visitMulExpr(AffineBinaryOpExpr expr) {
    Extent lhs = computeExtent(expr.getLHS(), vars);
    Extent rhs = computeExtent(expr.getRHS(), vars);
    std::array<int64_t, 3> values = {lhs.min * rhs.max, lhs.max * rhs.min,
                                     lhs.max * rhs.max};
    Extent result = {lhs.min * rhs.min, lhs.min * rhs.min};
    for (int64_t value : values) {
      if (value < result.min) {
        result.min = value;
      }
      if (value > result.max) {
        result.max = value;
      }
    }
    return result;
  }

  Extent visitAddExpr(AffineBinaryOpExpr expr) {
    Extent lhs = computeExtent(expr.getLHS(), vars);
    Extent rhs = computeExtent(expr.getRHS(), vars);
    return {lhs.min + rhs.min, lhs.max + rhs.max};
  }

  Extent visitDimExpr(AffineDimExpr expr) {
    unsigned pos = expr.getPosition();
    if (pos >= vars.size()) {
      throw std::runtime_error("Position exceeds the size of vars");
    }
    return vars[pos];
  }

  Extent visitConstantExpr(AffineConstantExpr expr) {
    int64_t value = expr.getValue();
    return {value, value};
  }

  Extent visitSymbolExpr(AffineSymbolExpr expr) {
    throw std::runtime_error("Unexpected affine expresssion: SymbolExpr.");
  }

  Extent visitCeilDivExpr(AffineBinaryOpExpr expr) {
    throw std::runtime_error("Unexpected affine expresssion: CeilDivExpr.");
  }

  Extent visitFloorDivExpr(AffineBinaryOpExpr expr) {
    throw std::runtime_error("Unexpected affine expresssion: FloorDivExpr.");
  }

  Extent visitModExpr(AffineBinaryOpExpr expr) {
    throw std::runtime_error("Unexpected affine expresssion: ModExpr.");
  }

private:
  ArrayRef<Extent> vars;
};

Extent computeExtent(AffineExpr expr, ArrayRef<Extent> vars) {
  ExtentComputer ec(vars);
  return ec.visit(expr);
}

} // namespace pmlc::util
