// Copyright 2021, Intel Corporation

#pragma once

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"

namespace pmlc::util {

struct Extent {
  int64_t min;
  int64_t max;
};

// Input an expression and the value range of all variables. Return the extent
// of the expression
Extent computeExtent(mlir::AffineExpr expr, mlir::ArrayRef<Extent> vars);

} // namespace pmlc::util
