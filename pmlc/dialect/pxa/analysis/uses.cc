// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/analysis/uses.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

IndirectValuesIterator &IndirectValuesIterator::operator++() {
  for (auto &use : curValue.getUses()) {
    if (auto yieldOp = dyn_cast<AffineYieldOp>(use.getOwner())) {
      auto value = yieldOp.getParentOp()->getResult(use.getOperandNumber());
      enqueueNext(value);
    } else if (auto reduceOp = dyn_cast<AffineReduceOp>(use.getOwner())) {
      enqueueNext(reduceOp.result());
    }
  }
  if (workQueue.empty()) {
    curValue = nullptr;
  } else {
    curValue = workQueue.front();
    workQueue.pop();
  }
  return *this;
}

void IndirectValuesIterator::enqueueNext(Value value) {
  if (!visited.count(value)) {
    visited.insert(value);
    workQueue.push(value);
  }
}

IndirectValuesRange getIndirectValues(Value value) {
  return {IndirectValuesIterator(value), IndirectValuesIterator()};
}

IndirectUsesIterator::IndirectUsesIterator(Value value)
    : inner(value), curIt(value.use_begin()) {
  nextValue();
}

IndirectUsesIterator &IndirectUsesIterator::operator++() {
  ++curIt;
  nextValue();
  return *this;
}

void IndirectUsesIterator::nextValue() {
  while (curIt == Value::use_iterator()) {
    ++inner;
    if (inner == IndirectValuesIterator()) {
      return;
    }
    curIt = inner.getValue().use_begin();
  }
}

IndirectUsesRange getIndirectUses(Value value) {
  return {IndirectUsesIterator(value), IndirectUsesIterator()};
}

IndirectAccessUsesIterator &IndirectAccessUsesIterator::operator++() {
  ++inner;
  skipNonAccess();
  return *this;
}

void IndirectAccessUsesIterator::skipNonAccess() {
  while (inner != IndirectUsesIterator()) {
    if (isa<AffineLoadOp>(inner->getOwner()) ||
        isa<AffineReduceOp>(inner->getOwner())) {
      break;
    }
    ++inner;
  }
}

IndirectAccessUsesRange getIndirectAccessUses(Value value) {
  return {IndirectAccessUsesIterator(value), IndirectAccessUsesIterator()};
}

} // namespace pmlc::dialect::pxa
