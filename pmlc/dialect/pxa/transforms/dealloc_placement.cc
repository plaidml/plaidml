// Copyright 2020 Intel Corporation

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/DebugStringHelper.h"

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

  using Callback = llvm::function_ref<void(unsigned)>;

  void runOnOperation() final {
    // Get the module
    ModuleOp op = getOperation();
    // Run all functions.  This could almost be a function pass, but init + fini
    // interact, which breaks the independence requirements
    op.walk([&](FuncOp fn) {
      if (fn.getName() == "init") {
        // If the function is named init, find fini
        auto finiFunc = op.lookupSymbol<FuncOp>("fini");
        if (!finiFunc) {
          fn.emitError() << "Init with no fini";
          signalPassFailure();
          return;
        }
        // Find fini's unpack op
        auto unpackOp = dyn_cast<stdx::UnpackOp>(finiFunc.begin()->begin());
        if (!unpackOp) {
          finiFunc.emitError() << "Fini must begin with unpack";
          signalPassFailure();
          return;
        }
        // Now, place deallocs on the init functions, moving escaping allocs to
        // be deallocated in fini
        OpBuilder builder(finiFunc.begin()->getTerminator());
        runOnFunction(fn, [&](unsigned i) {
          builder.create<memref::DeallocOp>(fn.getLoc(), unpackOp.getResult(i));
        });
      } else {
        // Place allocs, if any escape, it's an error
        runOnFunction(
            fn, [&](unsigned i) {
              fn.emitError()
                  << "Allocations escape via a pack for non-init function";
              signalPassFailure();
            });
      }
    });
  }

  void runOnFunction(FuncOp fn, Callback onPack) {
    // Place deallocation for AllocOp
    fn.walk([&](memref::AllocOp alloc) {
      IVLOG(3, "alloc: " << debugString(*alloc));
      Optional<Placement> placement =
          findPlacement(alloc.getResult(), alloc, alloc->getBlock(), onPack);
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

        Optional<Placement> placement = findPlacement(
            arg, &forOp.getBody()->front(), forOp.getBody(), onPack);
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
            /*reductions=*/AtomicRMWKind::assign,
            /*ranges=*/resType.getShape());

        Block *body = copyLoopOp.getBody();
        builder.setInsertionPointToStart(body);

        auto load = builder.create<pxa::PxaLoadOp>(forOp.getLoc(), init,
                                                   body->getArguments());

        AffineMap idMap = builder.getMultiDimIdentityMap(resType.getRank());
        auto reduce = builder.create<pxa::PxaReduceOp>(
            forOp.getLoc(), AtomicRMWKind::assign, load, newBuf, idMap,
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
                                    Block *allocBlock, Callback onPack) {
    OpOperand *lastUse = nullptr;
    for (OpOperand &itUse : getIndirectUses(ref)) {
      Operation *use = itUse.getOwner();
      IVLOG(3, "  use: " << debugString(*use));

      Operation *ancestor = allocBlock->findAncestorOpInBlock(*use);
      assert(ancestor && "use and alloc do not have a common ancestor");
      IVLOG(3, "  ancestor: " << debugString(*use));

      if (isa<ReturnOp>(ancestor)) {
        IVLOG(3, "  return");
        return None;
      }

      if (!ancestor->isBeforeInBlock(lastOp)) {
        lastOp = ancestor;
        lastUse = &itUse;
      }
    }

    IVLOG(3, "  last ancestor: " << debugString(*lastOp));

    if (auto packOp = dyn_cast<stdx::PackOp>(lastOp)) {
      IVLOG(3, "  pack op");
      // Alloc 'escapes' via a pack, call our callback to handle
      onPack(lastUse->getOperandNumber());
      return None;
    }

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
