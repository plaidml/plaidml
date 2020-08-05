#include "pmlc/dialect/pxa/analysis/affine_expr.h"

namespace mlir {

AffineValueExpr::AffineValueExpr(AffineExpr expr, ValueRange operands)
    : expr(expr), operands(operands.begin(), operands.end()) {}

AffineValueExpr::AffineValueExpr(MLIRContext *ctx, int64_t v) {
  expr = getAffineConstantExpr(v, ctx);
}

AffineValueExpr::AffineValueExpr(Value v) {
  operands.push_back(v);
  expr = getAffineDimExpr(0, v.getContext());
}

AffineValueExpr::AffineValueExpr(AffineValueMap map, unsigned idx)
    : expr(map.getResult(idx)),
      operands(map.getOperands().begin(), map.getOperands().end()) {}

AffineValueExpr AffineValueExpr::operator*(const AffineValueExpr &rhs) const {
  return AffineValueExpr(AffineExprKind::Mul, *this, rhs);
}

AffineValueExpr AffineValueExpr::operator*(int64_t rhs) const {
  auto cst = AffineValueExpr(expr.getContext(), rhs);
  return AffineValueExpr(AffineExprKind::Mul, *this, cst);
}

AffineValueExpr AffineValueExpr::operator+(const AffineValueExpr &rhs) const {
  return AffineValueExpr(AffineExprKind::Add, *this, rhs);
}

AffineValueExpr AffineValueExpr::operator+(int64_t rhs) const {
  auto cst = AffineValueExpr(expr.getContext(), rhs);
  return AffineValueExpr(AffineExprKind::Add, *this, cst);
}

AffineValueExpr AffineValueExpr::operator-(const AffineValueExpr &rhs) const {
  return *this + (rhs * -1);
}

AffineValueExpr AffineValueExpr::operator-(int64_t rhs) const {
  return *this - AffineValueExpr(expr.getContext(), rhs);
}

AffineExpr AffineValueExpr::getExpr() const { return expr; }

ArrayRef<Value> AffineValueExpr::getOperands() const { return operands; }

AffineValueExpr::AffineValueExpr(AffineExprKind kind, AffineValueExpr a,
                                 AffineValueExpr b)
    : operands(a.operands.begin(), a.operands.end()) {
  SmallVector<AffineExpr, 4> repl_b;
  for (auto v : b.operands) {
    auto it = std::find(operands.begin(), operands.end(), v);
    unsigned idx;
    if (it == operands.end()) {
      idx = operands.size();
      operands.push_back(v);
    } else {
      idx = it - operands.begin();
    }
    repl_b.push_back(getAffineDimExpr(idx, a.expr.getContext()));
  }
  auto new_b = b.expr.replaceDimsAndSymbols(repl_b, {});
  expr = getAffineBinaryOpExpr(kind, a.expr, new_b);
}

AffineValueMap jointValueMap(MLIRContext *ctx,
                             ArrayRef<AffineValueExpr> exprs) {
  DenseMap<Value, unsigned> jointSpace;
  for (const auto &expr : exprs) {
    for (Value v : expr.getOperands()) {
      if (!jointSpace.count(v)) {
        unsigned idx = jointSpace.size();
        jointSpace[v] = idx;
      }
    }
  }
  SmallVector<AffineExpr, 4> jointExprs;
  for (const auto &expr : exprs) {
    SmallVector<AffineExpr, 4> repl;
    for (Value v : expr.getOperands()) {
      repl.push_back(getAffineDimExpr(jointSpace[v], ctx));
    }
    jointExprs.push_back(expr.getExpr().replaceDimsAndSymbols(repl, {}));
  }
  SmallVector<Value, 4> jointOperands(jointSpace.size());
  for (const auto &kvp : jointSpace) {
    jointOperands[kvp.second] = kvp.first;
  }
  auto map = AffineMap::get(jointSpace.size(), 0, jointExprs, ctx);
  return AffineValueMap(map, jointOperands);
}

} // End namespace mlir
