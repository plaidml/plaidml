// Copyright 2019, Intel Corporation

#include "pmlc/util/slice.h"

#include <stack>
#include <unordered_set>
#include <utility>

#include "llvm/ADT/SetVector.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace pmlc {
namespace util {

namespace {

class BackwardSliceImpl {
 public:
  std::vector<mlir::Value*> slice;

  BackwardSliceImpl(const llvm::SetVector<mlir::Value*>& values, bool enter_regions, TransitiveFilter filter) {
    for (auto it = values.rbegin(); it != values.rend(); it++) {
      Push(*it, [](mlir::Value*) { return true; });
    }
    std::unordered_set<const mlir::Value*> seen;
    while (stack.size()) {
      auto entry = stack.top();
      stack.pop();
      auto value = entry.first;
      if (entry.second) {
        slice.emplace_back(value);
      } else if (!seen.count(value)) {
        seen.insert(value);
        stack.push(std::make_pair(value, true));
        auto op = value->getDefiningOp();
        if (op) {
          Push(op, enter_regions, filter);
        }
      }
    }
  }

 private:
  void Push(mlir::Value* value, TransitiveFilter filter) {
    if (filter(value)) {
      stack.push(std::make_pair(value, false));
    }
  }

  void Push(mlir::Operation* op, bool enter_regions, TransitiveFilter filter) {
    for (auto operand : op->getOperands()) {
      Push(operand, filter);
    }
    if (enter_regions) {
      for (auto& region : op->getRegions()) {
        for (auto& block : region) {
          for (auto it = block.rbegin(); it != block.rend(); it++) {
            Push(&*it, enter_regions, filter);
          }
        }
      }
    }
  }

  std::stack<std::pair<mlir::Value*, bool>> stack;
};

}  // namespace

std::vector<mlir::Value*> getBackwardSlice(       //
    const llvm::SetVector<mlir::Value*>& values,  //
    bool enter_regions,                           //
    TransitiveFilter filter) {
  BackwardSliceImpl impl(values, enter_regions, filter);
  return impl.slice;
}

}  // namespace util
}  // namespace pmlc
