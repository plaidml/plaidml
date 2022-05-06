// Copyright 2020 Intel Corporation

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::pxa {

struct DeallocPlacementPass
    : public DeallocPlacementBase<DeallocPlacementPass> {

  struct Placement {
    Operation *nextOp;
    OpOperand *lastUse;
  };

  void runOnOperation() final {
    ModuleOp op = getOperation();
    op.walk([&](FuncOp fn) { runOnFunction(fn); });
  }

  void runOnFunction(func::FuncOp fn) {
    // Place deallocation for AllocOp
    fn.walk([&](memref::AllocOp alloc) {
      IVLOG(3, "alloc: " << debugString(*alloc));
      Optional<Placement> placement =
          findPlacement(alloc.getResult(), alloc, alloc->getBlock());
      if (!placement)
        return;

      OpBuilder builder(placement->nextOp);
      builder.create<memref::DeallocOp>(alloc.getLoc(), alloc);
    });

    // Place deallocation for `scf.for` iter_args
    fn.walk([&](scf::ForOp forOp) {
      IVLOG(3, "scf.for: " << debugString(*forOp));
      for (auto it : llvm::enumerate(
               llvm::zip(forOp.getRegionIterArgs(), forOp.getIterOperands()))) {
        BlockArgument arg;
        Value init;
        std::tie(arg, init) = it.value();

        Optional<Placement> placement =
            findPlacement(arg, &forOp.getBody()->front(), forOp.getBody());
        if (!placement)
          continue;

        OpBuilder builder(placement->nextOp);
        builder.create<memref::DeallocOp>(forOp.getLoc(),
                                          placement->lastUse->get());

        // Here we need to process scf.for argument, i.e., copy the initial
        // argument
        builder.setInsertionPoint(forOp);
        MemRefType resType = init.getType().cast<MemRefType>();

        // Build the buffer for the new tensor
        auto newBuf = builder.create<memref::AllocOp>(forOp.getLoc(), resType);

        // Build a element-wise for to copy the initial value to the new buffer
        auto copyLoopOp = builder.create<AffineParallelOp>(
            forOp.getLoc(),
            /*resultTypes=*/resType,
            /*reductions=*/arith::AtomicRMWKind::assign,
            /*ranges=*/resType.getShape());

        Block *body = copyLoopOp.getBody();
        builder.setInsertionPointToStart(body);

        auto load = builder.create<pxa::PxaLoadOp>(forOp.getLoc(), init,
                                                   body->getArguments());

        AffineMap idMap = builder.getMultiDimIdentityMap(resType.getRank());
        auto reduce = builder.create<pxa::PxaReduceOp>(
            forOp.getLoc(), arith::AtomicRMWKind::assign, load, newBuf, idMap,
            builder.getBlock()->getArguments());

        builder.create<AffineYieldOp>(forOp.getLoc(),
                                      ValueRange{reduce.result()});

        // Replace the initial argument with the result of new loop
        forOp.setOperand(forOp.getNumControlOperands() + it.index(),
                         copyLoopOp.getResult(0));
      }
    });
  }

  Optional<Placement> findPlacement(Value ref, Operation *lastOp,
                                    Block *allocBlock) {
    OpOperand *lastUse = nullptr;
    for (OpOperand &itUse : getIndirectUses(ref)) {
      Operation *use = itUse.getOwner();
      IVLOG(3, "  use: " << debugString(*use));

      Operation *ancestor = allocBlock->findAncestorOpInBlock(*use);
      assert(ancestor && "use and alloc do not have a common ancestor");
      IVLOG(3, "  ancestor: " << debugString(*use));

      if (isa<func::ReturnOp>(ancestor)) {
        IVLOG(3, "  return");
        return None;
      }

      if (!ancestor->isBeforeInBlock(lastOp)) {
        lastOp = ancestor;
        lastUse = &itUse;
      }
    }

    IVLOG(3, "  last ancestor: " << debugString(*lastOp));
    Operation *nextOp = lastOp->getNextNode();
    if (!nextOp) {
      IVLOG(3, "  terminator");
      return None;
    }

    IVLOG(3, "  next operation: " << debugString(*nextOp));
    return Placement{nextOp, lastUse};
  }
};

std::unique_ptr<mlir::Pass> createDeallocPlacementPass() {
  return std::make_unique<DeallocPlacementPass>();
}

} // namespace pmlc::dialect::pxa
