// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/Dialect/Affine/IR/AffineValueMap.h"

namespace mlir {

class AffineValueExpr {
public:
  AffineValueExpr(AffineExpr expr, ValueRange operands);
  AffineValueExpr(MLIRContext *ctx, int64_t v);
  explicit AffineValueExpr(Value v);
  AffineValueExpr(AffineValueMap map, unsigned idx);
  AffineValueExpr operator*(const AffineValueExpr &rhs) const;
  AffineValueExpr operator*(int64_t rhs) const;
  AffineValueExpr operator+(const AffineValueExpr &rhs) const;
  AffineValueExpr operator+(int64_t rhs) const;
  AffineValueExpr operator-(const AffineValueExpr &rhs) const;
  AffineValueExpr operator-(int64_t rhs) const;
  AffineExpr getExpr() const;
  ArrayRef<Value> getOperands() const;

private:
  AffineValueExpr(AffineExprKind kind, AffineValueExpr a, AffineValueExpr b);
  AffineExpr expr;
  SmallVector<Value, 4> operands;
};

AffineValueMap jointValueMap(MLIRContext *ctx, ArrayRef<AffineValueExpr> exprs);

} // End namespace mlir
