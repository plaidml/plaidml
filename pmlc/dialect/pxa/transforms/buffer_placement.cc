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
        Operation *use = itUse.getOwner();
        IVLOG(3, "  use: " << debugString(*use));

        while (use->getBlock() != allocBlock) {
          use = use->getParentOp();
          assert(use && "use does not have a common ancestor");
          IVLOG(3, "  parent: " << debugString(*use));
        }

        if (isa<ReturnOp>(use)) {
          IVLOG(3, "  return");
          return;
        }

        if (!use->isBeforeInBlock(lastOp)) {
          lastOp = use;
        }
      }
      IVLOG(3, "  last: " << debugString(*lastOp));

      Operation *nextOp = lastOp->getNextNode();
      if (!nextOp) {
        IVLOG(3, "  terminator");
        return;
      }

      IVLOG(3, "  nextOp: " << debugString(*nextOp));
      OpBuilder builder(nextOp);
      builder.create<DeallocOp>(alloc.getLoc(), alloc);
    });
  }
};

std::unique_ptr<mlir::Pass> createBufferPlacementPass() {
  return std::make_unique<BufferPlacementPass>();
}

} // namespace pmlc::dialect::pxa
