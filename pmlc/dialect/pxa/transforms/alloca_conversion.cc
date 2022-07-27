// Copyright 2020 Intel Corporation
#include <list>

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

struct AllocaConversionPass
    : public AllocaConversionBase<AllocaConversionPass> {
  void runOnOperation() final {
    func::FuncOp f = getOperation();
    std::list<Operation *> toMove;

    f.walk([&](memref::AllocOp allocOp) {
      // Find the nearest common ancestor for the users of the alloc op.
      auto *op = allocOp.getOperation();
      if (isa<AffineParallelOp>(op->getBlock()->getParentOp()->getParentOp())) {
        toMove.push_back(op);
      }
    });

    // Mark and sweep to avoid redundant analysis.
    for (auto &kvp : toMove) {
      Operation *allocOperation = kvp;
      auto allocOp = cast<memref::AllocOp>(allocOperation);

      OpBuilder builder(allocOp);
      auto allocaOp = builder.create<mlir::memref::AllocaOp>(
          allocOp.getLoc(), allocOp.getType().cast<MemRefType>(),
          allocOp->getOperands());
      allocOp.replaceAllUsesWith(&*allocaOp);
      allocOp.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createAllocaConversionPass() {
  return std::make_unique<AllocaConversionPass>();
}

} // namespace pmlc::dialect::pxa
