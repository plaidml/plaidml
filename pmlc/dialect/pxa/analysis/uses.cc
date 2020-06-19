// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/analysis/uses.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Support/DebugStringHelper.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

IndirectDefsIterator &IndirectDefsIterator::operator++() {
  assert(curValue && "Should not do ++ on end");
  // Walk all uses and find the unique non-read user
  unsigned nextOperand = 0;
  Operation *nextOp = nullptr;
  for (auto &use : curValue.getUses()) {
    if (isa<AffineLoadOp>(use.getOwner())) {
      continue;
    }
    assert(!nextOp && "PXA memref-states should have only one non-read user");
    nextOp = use.getOwner();
    nextOperand = use.getOperandNumber();
  }
  // If no next op, set to done
  if (!nextOp) {
    curValue = Value();
    return *this;
  }
  if (auto yieldOp = dyn_cast<AffineYieldOp>(nextOp)) {
    curValue = yieldOp.getParentOp()->getResult(nextOperand);
  } else if (auto reduceOp = dyn_cast<AffineReduceOp>(nextOp)) {
    curValue = reduceOp.result();
  } else {
    llvm_unreachable("All uses of pxa mem-ref state should be reads, "
                     "yields, reduces, or returns");
  }
  return *this;
}

void IndirectUsesIterator::enqueueNext(Value value) {
  if (!value.use_empty() && !visited.count(value)) {
    visited.insert(value);
    workQueue.push(value.use_begin());
  }
}

IndirectUsesIterator &IndirectUsesIterator::operator++() {
  if (auto yieldOp = dyn_cast<AffineYieldOp>(curIt->getOwner())) {
    auto value = yieldOp.getParentOp()->getResult(curIt->getOperandNumber());
    enqueueNext(value);
  } else if (auto reduceOp = dyn_cast<AffineReduceOp>(curIt->getOwner())) {
    enqueueNext(reduceOp.result());
  }
  curIt++;
  if (curIt == Value::use_iterator() && !workQueue.empty()) {
    curIt = workQueue.front();
    workQueue.pop();
  }
  return *this;
}

AccessIndirectUsesIterator &AccessIndirectUsesIterator::operator++() {
  ++inner;
  skipNonAccess();
  return *this;
}

void AccessIndirectUsesIterator::skipNonAccess() {
  while (inner != IndirectUsesIterator()) {
    if (isa<AffineLoadOp>(inner->getOwner()) ||
        isa<AffineReduceOp>(inner->getOwner())) {
      break;
    }
    ++inner;
  }
}

} // namespace pmlc::dialect::pxa
