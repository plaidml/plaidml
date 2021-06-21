// Copyright 2020, Intel Corporation

#include "pmlc/dialect/pxa/ir/interfaces.h"

#include "pmlc/dialect/pxa/ir/interfaces.cc.inc"

namespace mlir {

OpOperandVector::operator SmallVector<Value>() {
  SmallVector<Value> result;
  result.reserve(this->size());
  for (OpOperand *opOperand : *this)
    result.push_back(opOperand->get());
  return result;
}

} // namespace mlir
