#pragma once

#include <queue>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/iterator.h"

#include "pmlc/dialect/pxa/ir/ops.h"

namespace pmlc::dialect::pxa {

// Trace through any parallel fors to find the previous writer operation for a
// given value.
mlir::Operation *getPrevWriter(mlir::Value value);

mlir::Value getPrevIndirectDef(mlir::OpResult def);
mlir::Value getNextIndirectUse(mlir::OpOperand &use);

mlir::Value getIndirectDef(mlir::Value value);
mlir::Value getIndirectDefOutsideScope(mlir::Value value,
                                       mlir::Operation *scope);

class IndirectValuesIterator
    : public llvm::iterator_facade_base<
          IndirectValuesIterator, std::forward_iterator_tag, mlir::Value> {
public:
  IndirectValuesIterator() {}
  explicit IndirectValuesIterator(mlir::Value value) : curValue(value) {}

  IndirectValuesIterator &
  operator=(const IndirectValuesIterator &other) = default;

  bool operator==(const IndirectValuesIterator &rhs) const {
    return curValue == rhs.curValue;
  }

  mlir::Value getValue() const { return curValue; }
  mlir::Value operator*() const { return curValue; }

  IndirectValuesIterator &operator++();
  IndirectValuesIterator operator++(int) {
    IndirectValuesIterator cur = *this;
    ++(*this);
    return cur;
  }

private:
  void enqueueNext(mlir::Value value);

private:
  // The current value.
  mlir::Value curValue;
  // The next values to process.
  std::queue<mlir::Value> workQueue;
  // Avoid duplicate visitations.
  llvm::DenseSet<mlir::Value> visited;
};

using IndirectValuesRange = llvm::iterator_range<IndirectValuesIterator>;
IndirectValuesRange getIndirectValues(mlir::Value value);

class IndirectUsesIterator
    : public llvm::iterator_facade_base<
          IndirectUsesIterator, std::forward_iterator_tag, mlir::OpOperand> {
public:
  IndirectUsesIterator() {}
  explicit IndirectUsesIterator(mlir::Value value);

  IndirectUsesIterator &operator=(const IndirectUsesIterator &other) = default;

  bool operator==(const IndirectUsesIterator &rhs) const {
    return std::tie(inner, curIt) == std::tie(rhs.inner, rhs.curIt);
  }

  const mlir::OpOperand &operator*() const { return *curIt; }
  mlir::OpOperand &operator*() { return *curIt; }

  IndirectUsesIterator &operator++();
  IndirectUsesIterator operator++(int) {
    IndirectUsesIterator cur = *this;
    ++(*this);
    return cur;
  }

private:
  void nextValue();

private:
  // The current value iterator.
  IndirectValuesIterator inner;
  // The use iterator.
  mlir::Value::use_iterator curIt;
};

using IndirectUsesRange = llvm::iterator_range<IndirectUsesIterator>;
IndirectUsesRange getIndirectUses(mlir::Value value);

// Subsets all uses to only acceses (skipping yield/return)
class IndirectAccessUsesIterator
    : public llvm::iterator_facade_base<IndirectAccessUsesIterator,
                                        std::forward_iterator_tag,
                                        mlir::OpOperand> {
public:
  IndirectAccessUsesIterator() {}

  explicit IndirectAccessUsesIterator(mlir::Value value) : inner(value) {
    skipNonAccess();
  }

  IndirectAccessUsesIterator &
  operator=(const IndirectAccessUsesIterator &other) = default;

  bool operator==(const IndirectAccessUsesIterator &rhs) const {
    return inner == rhs.inner;
  }

  const mlir::OpOperand &operator*() const { return *inner; }
  mlir::OpOperand &operator*() { return *inner; }

  IndirectAccessUsesIterator &operator++();
  IndirectAccessUsesIterator operator++(int) {
    IndirectAccessUsesIterator cur = *this;
    ++(*this);
    return cur;
  }

private:
  void skipNonAccess();

  IndirectUsesIterator inner;
};

using IndirectAccessUsesRange =
    llvm::iterator_range<IndirectAccessUsesIterator>;
IndirectAccessUsesRange getIndirectAccessUses(mlir::Value value);

} // namespace pmlc::dialect::pxa
