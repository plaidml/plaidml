// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::pxa {

namespace {

class LocalizeAnalysis {
public:
  explicit LocalizeAnalysis(Operation *op) : domInfo(op) {}

  Block *findNearestCommonAncestor(ArrayRef<Operation *> ops) {
    if (ops.empty())
      return nullptr;
    auto *top = ops.front()->getBlock();
    for (auto *op : ops.drop_front()) {
      auto *block = op->getBlock();
      top = domInfo.findNearestCommonDominator(top, block);
    }
    return top;
  }

private:
  DominanceInfo domInfo;
};

struct LocalizePass : public LocalizeBase<LocalizePass> {
  void runOnOperation() final {
    func::FuncOp f = getOperation();
    DenseMap<Operation *, Block *> toMove;
    auto &analysis = getAnalysis<LocalizeAnalysis>();

    f.walk([&](memref::AllocOp allocOp) {
      // Collect all the indirect users of the alloc op.
      SmallVector<Operation *, 4> opGroup;
      for (auto &use : getIndirectUses(allocOp)) {
        opGroup.push_back(use.getOwner());
      }

      // Find the nearest common ancestor for the users of the alloc op.
      auto *target = analysis.findNearestCommonAncestor(opGroup);
      auto *op = allocOp.getOperation();
      if (target != op->getBlock()) {
        toMove[op] = target;
      }
    });

    // Mark and sweep to avoid redundant analysis.
    for (auto &kvp : toMove) {
      kvp.first->moveBefore(&kvp.second->front());
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLocalizePass() {
  return std::make_unique<LocalizePass>();
}

} // namespace pmlc::dialect::pxa
