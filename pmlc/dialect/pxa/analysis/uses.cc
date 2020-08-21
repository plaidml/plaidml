// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/analysis/uses.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Support/DebugStringHelper.h"
#include "pmlc/dialect/stdx/ir/ops.h"

#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

Operation *getOriginalDef(Value val) {
  auto opRes = val.cast<mlir::OpResult>();
  while (true) {
    if (auto ap = mlir::dyn_cast<AffineParallelOp>(opRes.getOwner())) {
      auto ret = mlir::cast<AffineYieldOp>(ap.getBody()->getTerminator());
      auto src = ret.getOperand(opRes.getResultNumber());
      opRes = src.cast<mlir::OpResult>();
    } else if (auto iop = dyn_cast<AffineIfOp>(opRes.getOwner())) {
      auto ret = mlir::cast<AffineYieldOp>(iop.getThenBlock()->getTerminator());
      auto src = ret.getOperand(opRes.getResultNumber());
      opRes = src.cast<mlir::OpResult>();
    } else {
      break;
    }
  }
  return opRes.getOwner();
}

IndirectValuesIterator &IndirectValuesIterator::operator++() {
  for (auto &use : curValue.getUses()) {
    if (auto yieldOp = dyn_cast<AffineYieldOp>(use.getOwner())) {
      auto value = yieldOp.getParentOp()->getResult(use.getOperandNumber());
      enqueueNext(value);
    } else if (auto reduceOp = dyn_cast<PxaReduceOp>(use.getOwner())) {
      enqueueNext(reduceOp.result());
    } else if (auto vecReduceOp = dyn_cast<PxaVectorReduceOp>(use.getOwner())) {
      enqueueNext(vecReduceOp.result());
    } else if (auto gemmOp = dyn_cast<AffineGemmOp>(use.getOwner())) {
      if (gemmOp.getOperand(use.getOperandNumber()) == gemmOp.c()) {
        enqueueNext(gemmOp.out());
      }
    } else if (auto prngOp = dyn_cast<PrngOp>(use.getOwner())) {
      if (prngOp.getOperand(use.getOperandNumber()) == prngOp.tensor()) {
        enqueueNext(prngOp.result_tensor());
      } else if (prngOp.getOperand(use.getOperandNumber()) ==
                 prngOp.new_state()) {
        enqueueNext(prngOp.result_state());
      }
    } else if (auto reshapeOp =
                   dyn_cast<pmlc::dialect::stdx::ReshapeOp>(use.getOwner())) {
      enqueueNext(reshapeOp.result());
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
    if (isa<PxaLoadOp>(inner->getOwner()) ||
        isa<PxaReduceOp>(inner->getOwner()) ||
        isa<PxaVectorLoadOp>(inner->getOwner()) ||
        isa<PxaVectorReduceOp>(inner->getOwner())) {
      break;
    }
    ++inner;
  }
}

IndirectAccessUsesRange getIndirectAccessUses(Value value) {
  return {IndirectAccessUsesIterator(value), IndirectAccessUsesIterator()};
}

} // namespace pmlc::dialect::pxa
