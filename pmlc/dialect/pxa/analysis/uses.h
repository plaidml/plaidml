#pragma once

#include <queue>
#include <set>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/iterator.h"

#include "pmlc/dialect/pxa/ir/ops.h"

namespace pmlc::dialect::pxa {

class IndirectDefsIterator
    : public llvm::iterator_facade_base<
          IndirectDefsIterator, std::forward_iterator_tag, mlir::Value> {
public:
  IndirectDefsIterator() {}
  explicit IndirectDefsIterator(Value value) : curValue(value) {}

  IndirectDefsIterator &operator=(const IndirectDefsIterator &other) = default;

  bool operator==(const IndirectDefsIterator &rhs) const {
    return curValue == rhs.curValue;
  }

  const mlir::Value &operator*() const { return curValue; }
  mlir::Value &operator*() { return curValue; }

  IndirectDefsIterator &operator++();

private:
  // The current def being considered
  Value curValue;
};

class IndirectUsesIterator
    : public llvm::iterator_facade_base<
          IndirectUsesIterator, std::forward_iterator_tag, mlir::OpOperand> {
public:
  IndirectUsesIterator() {}
  explicit IndirectUsesIterator(Value value) : curIt(value.use_begin()) {}

  IndirectUsesIterator &operator=(const IndirectUsesIterator &other) = default;

  bool operator==(const IndirectUsesIterator &rhs) const {
    return curIt == rhs.curIt;
  }

  const mlir::OpOperand &operator*() const { return *curIt; }
  mlir::OpOperand &operator*() { return *curIt; }

  IndirectUsesIterator &operator++();

private:
  void enqueueNext(Value value);

private:
  // The position in the walk
  Value::use_iterator curIt;
  // The next iterators to follow once we finish with the current iterator.
  std::queue<Value::use_iterator> workQueue;
  // Avoid duplicate visitations.
  llvm::DenseSet<Value> visited;
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

  AccessIndirectUsesIterator &operator++();

private:
  void skipNonAccess();

  IndirectUsesIterator inner;
};

class IndirectDefs {
public:
  explicit IndirectDefs(Value value) : value(value) {}
  IndirectDefsIterator begin() { return IndirectDefsIterator(value); }
  IndirectDefsIterator end() { return IndirectDefsIterator(); }

private:
  Value value;
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

} // namespace pmlc::dialect::pxa
