#pragma once

#include "pmlc/dialect/pxa/ir/ops.h"
#include "llvm/ADT/iterator.h"

namespace pmlc::dialect::pxa {

class IndirectUsesIterator
    : public llvm::iterator_facade_base<
          IndirectUsesIterator, std::forward_iterator_tag, mlir::OpOperand> {
public:
  IndirectUsesIterator() : curValue(nullptr), curIt(), next(nullptr) {}

  explicit IndirectUsesIterator(Value value)
      : curValue(value), curIt(value.use_begin()), next(nullptr) {
    skipNonRead();
  }

  IndirectUsesIterator &operator=(const IndirectUsesIterator &other) = default;
  bool operator==(const IndirectUsesIterator &rhs) const {
    return curValue == rhs.curValue && curIt == rhs.curIt && next == rhs.next;
  }

  const mlir::OpOperand &operator*() const {
    return curIt == curValue.use_end() ? *next : *curIt;
  }

  mlir::OpOperand &operator*() {
    return curIt == curValue.use_end() ? *next : *curIt;
  }

  IndirectUsesIterator &operator++() {
    assert(curValue && "Invalid curValue");
    if (curIt != curValue.use_end()) {
      // Walking over a read use
      curIt++;
      skipNonRead();
    }
    // We've finished the readers
    if (curIt == curValue.use_end()) {
      // Maybe we are all done?
      if (!next || mlir::isa<mlir::ReturnOp>(next->getOwner())) {
        curValue = nullptr;
        curIt = Value::use_iterator();
        next = nullptr;
        return *this;
      }
      // Otherwise, move to next value + reset it + next
      if (auto yieldOp = mlir::dyn_cast<AffineYieldOp>(next->getOwner())) {
        curValue = yieldOp.getParentOp()->getResult(next->getOperandNumber());
      } else if (auto reduceOp =
                     mlir::dyn_cast<AffineReduceOp>(next->getOwner())) {
        curValue = reduceOp.result();
      } else {
        llvm_unreachable("All uses of pxa mem-ref state should be reads, "
                         "yields, or reduces or returns");
      }
      curIt = curValue.use_begin();
      next = nullptr;
      skipNonRead();
    }
    return *this;
  }

private:
  void skipNonRead() {
    while (curIt != curValue.use_end() &&
           !mlir::isa<mlir::AffineLoadOp>(curIt->getOwner())) {
      assert(!next && "PXA memref-states should have only one non-read user");
      next = &*curIt;
      curIt++;
    }
  }
  // The current value being walked
  Value curValue;
  // The position in the walk
  Value::use_iterator curIt;
  // The next operation to follow once we finish with all read uses
  mlir::OpOperand *next;
};

// Subsets all uses to only acceses (skipping yield/return)
class AccessIndirectUsesIterator
    : public llvm::iterator_facade_base<AccessIndirectUsesIterator,
                                        std::forward_iterator_tag,
                                        mlir::OpOperand> {
public:
  AccessIndirectUsesIterator() {}

  explicit AccessIndirectUsesIterator(Value value) : inner(value) {
    skipNonAccess();
  }

  AccessIndirectUsesIterator &
  operator=(const AccessIndirectUsesIterator &other) = default;

  bool operator==(const AccessIndirectUsesIterator &rhs) const {
    return inner == rhs.inner;
  }

  const mlir::OpOperand &operator*() const { return *inner; }

  mlir::OpOperand &operator*() { return *inner; }

  AccessIndirectUsesIterator &operator++() {
    ++inner;
    skipNonAccess();
    return *this;
  }

private:
  void skipNonAccess() {
    while (inner != IndirectUsesIterator()) {
      if (mlir::isa<mlir::AffineLoadOp>(inner->getOwner()) ||
          mlir::isa<AffineReduceOp>(inner->getOwner())) {
        break;
      }
      ++inner;
    }
  }
  IndirectUsesIterator inner;
};

class IndirectUses {
public:
  explicit IndirectUses(Value value) : value(value) {}
  IndirectUsesIterator begin() { return IndirectUsesIterator(value); }
  IndirectUsesIterator end() { return IndirectUsesIterator(); }

private:
  Value value;
};

class AccessIndirectUses {
public:
  explicit AccessIndirectUses(Value value) : value(value) {}

  AccessIndirectUsesIterator begin() {
    return AccessIndirectUsesIterator(value);
  }

  AccessIndirectUsesIterator end() { return AccessIndirectUsesIterator(); }

private:
  Value value;
};

} // End namespace pmlc::dialect::pxa
