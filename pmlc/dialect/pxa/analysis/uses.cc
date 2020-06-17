// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/analysis/uses.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

IndirectUsesIterator &IndirectUsesIterator::operator++() {
  assert(curValue && "Invalid curValue");
  if (curIt != curValue.use_end()) {
    // Walking over a read use
    curIt++;
    skipNonRead();
  }
  // We've finished the readers
  if (curIt == curValue.use_end()) {
    // Maybe we are all done?
    if (!next || isa<ReturnOp>(next->getOwner())) {
      curValue = nullptr;
      curIt = Value::use_iterator();
      next = nullptr;
      return *this;
    }
    // Otherwise, move to next value + reset it + next
    if (auto yieldOp = dyn_cast<AffineYieldOp>(next->getOwner())) {
      curValue = yieldOp.getParentOp()->getResult(next->getOperandNumber());
    } else if (auto reduceOp = dyn_cast<AffineReduceOp>(next->getOwner())) {
      curValue = reduceOp.result();
    } else {
      llvm_unreachable("All uses of pxa mem-ref state should be reads, "
                       "yields, reduces, or returns");
    }
    curIt = curValue.use_begin();
    next = nullptr;
    skipNonRead();
  }
  return *this;
}

void IndirectUsesIterator::skipNonRead() {
  while (curIt != curValue.use_end() && !isa<AffineLoadOp>(curIt->getOwner())) {
    assert(!next && "PXA memref-states should have only one non-read user");
    next = &*curIt;
    curIt++;
  }
}

} // namespace pmlc::dialect::pxa
