// Copyright 2020 Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::pxa {

template <typename Iterator, typename Range>
static Iterator getLast(Range range) {
  auto it = range.begin();
  auto itEnd = range.end();
  auto itNext = it;
  while (itNext != itEnd) {
    it = itNext;
    ++itNext;
  }
  return it;
}

struct BufferPlacementPass : public BufferPlacementBase<BufferPlacementPass> {
  void runOnFunction() final {
    auto fn = getFunction();
    fn.walk([&](AllocOp alloc) {
      IVLOG(1, "alloc: " << debugString(*alloc));
      auto itLast = getLast<IndirectUsesIterator>(getIndirectUses(alloc));
      Operation *lastOp = itLast->getOwner();
      IVLOG(1, "  last: " << debugString(*lastOp));

      if (isa<ReturnOp>(lastOp)) {
        IVLOG(1, "  return");
        return;
      }

      Block *allocBlock = alloc.getOperation()->getBlock();
      while (lastOp->getBlock() != allocBlock) {
        lastOp = lastOp->getParentOp();
        assert(lastOp && "last op does not have a common ancestor");
        IVLOG(1, "  up, last: " << debugString(*lastOp));
      }

      Operation *nextOp = lastOp->getNextNode();
      if (!nextOp) {
        IVLOG(1, "  terminator");
        return;
      }

      IVLOG(1, "  nextOp: " << debugString(*nextOp));
      OpBuilder builder(nextOp);
      builder.create<DeallocOp>(alloc.getLoc(), alloc);
    });
  }
};

std::unique_ptr<mlir::Pass> createBufferPlacementPass() {
  return std::make_unique<BufferPlacementPass>();
}

} // namespace pmlc::dialect::pxa
