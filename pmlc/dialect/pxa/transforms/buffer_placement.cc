// Copyright 2020 Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::pxa {

struct BufferPlacementPass : public BufferPlacementBase<BufferPlacementPass> {
  void runOnFunction() final {
    auto fn = getFunction();
    fn.walk([&](AllocOp alloc) {
      IVLOG(3, "alloc: " << debugString(*alloc));

      Operation *lastOp = alloc;
      Block *allocBlock = alloc.getOperation()->getBlock();
      for (auto &itUse : getIndirectUses(alloc)) {
        auto use = itUse.getOwner();
        IVLOG(3, "  use: " << debugString(*use));

        auto ancestor = allocBlock->findAncestorOpInBlock(*use);
        assert(ancestor && "use and alloc do not have a common ancestor");
        IVLOG(3, "  ancestor: " << debugString(*use));

        if (isa<ReturnOp>(ancestor)) {
          IVLOG(3, "  return");
          return;
        }

        if (!ancestor->isBeforeInBlock(lastOp)) {
          lastOp = ancestor;
        }
      }
      IVLOG(3, "  last ancestor: " << debugString(*lastOp));

      Operation *nextOp = lastOp->getNextNode();
      if (!nextOp) {
        IVLOG(3, "  terminator");
        return;
      }

      IVLOG(3, "  next operation: " << debugString(*nextOp));
      OpBuilder builder(nextOp);
      builder.create<DeallocOp>(alloc.getLoc(), alloc);
    });
  }
};

std::unique_ptr<mlir::Pass> createBufferPlacementPass() {
  return std::make_unique<BufferPlacementPass>();
}

} // namespace pmlc::dialect::pxa
