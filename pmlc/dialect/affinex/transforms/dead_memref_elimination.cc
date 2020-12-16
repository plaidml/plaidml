// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "pmlc/dialect/affinex/transforms/pass_detail.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::affinex {

struct AffinexDeadMemRefElimination
    : public AffinexDeadMemRefEliminationBase<AffinexDeadMemRefElimination> {

  void runOnFunction() override {
    llvm::SmallVector<Operation *, 8> opsToErase;
    getFunction().walk([&](AllocOp alloc) {
      auto memref = alloc.getResult();
      for (Operation *user : memref.getUsers()) {
        if (isa<AffineWriteOpInterface, DeallocOp>(user)) {
          opsToErase.push_back(user);
        } else {
          opsToErase.clear();
          return;
        }
      }
      opsToErase.push_back(alloc.getOperation());
    });
    for (auto *op : opsToErase) {
      op->erase();
    }
  }
};

std::unique_ptr<Pass> createAffinexDeadMemRefElimination() {
  return std::make_unique<AffinexDeadMemRefElimination>();
}
} // namespace pmlc::dialect::affinex
