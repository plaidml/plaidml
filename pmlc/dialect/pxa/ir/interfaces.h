// Copyright 2020, Intel Corporation

#pragma once

#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {

struct OpOperandVector : public SmallVector<OpOperand *> {
  operator SmallVector<Value>();
};

} // namespace mlir

#include "pmlc/dialect/pxa/ir/interfaces.h.inc"
