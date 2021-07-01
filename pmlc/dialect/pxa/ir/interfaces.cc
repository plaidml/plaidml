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

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

bool PxaMemAccessOperand::isRead() const {
  auto op = cast<PxaGenericOpInterface>(getOperation());
  return op.isReadOperand(opOperand);
}

bool PxaMemAccessOperand::isWrite() const {
  auto op = cast<PxaGenericOpInterface>(getOperation());
  return op.isWriteOperand(opOperand);
}

SmallVector<StrideRange> PxaMemAccessOperand::getInternalRanges() const {
  auto op = cast<PxaGenericOpInterface>(getOperation());
  return op.getTiedInternalRanges(opOperand);
}

AffineValueMap PxaMemAccessOperand::getAffineValueMap() const {
  auto op = cast<PxaGenericOpInterface>(getOperation());
  return op.getTiedAffineValueMap(opOperand);
}

} // namespace pmlc::dialect::pxa
