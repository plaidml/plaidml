// Copyright 2020 Intel Corporation

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
        // be dealloced in fini
        OpBuilder deallocBuilder(finiFunc.begin()->getTerminator());
        runOnFunction(fn, [&](unsigned i) {
          deallocBuilder.create<DeallocOp>(fn.getLoc(), unpackOp.getResult(i));
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

  template <typename Callback> void runOnFunction(FuncOp fn, Callback onPack) {
    fn.walk([&](AllocOp alloc) {
      IVLOG(3, "alloc: " << debugString(*alloc));
      placeDealloc(alloc.getResult(), alloc, alloc.getOperation()->getBlock(),
                   -1, onPack);
    });
    fn.walk([&](scf::ForOp forOp) {
      auto args = forOp.getRegionIterArgs();
      for (unsigned i = 0; i < args.size(); ++i) {
        IVLOG(3, "for argument: " << debugString(args[i]));
        placeDealloc(args[i].cast<Value>(), &forOp.getBody()->front(),
                     forOp.getBody(), i, onPack);
      }
    });
  }

  template <typename Callback>
  void placeDealloc(Value ref, Operation *allocOp, Block *allocBlock,
                    int argNumber, Callback onPack) {
    Operation *lastOp = allocOp;
    OpOperand *lastUse = nullptr;
    for (auto &itUse : getIndirectUses(ref)) {
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
        lastUse = &itUse;
      }
    }
    IVLOG(3, "  last ancestor: " << debugString(*lastOp));
    if (auto packOp = dyn_cast<stdx::PackOp>(lastOp)) {
      IVLOG(3, "  pack op");
      // Alloc 'escapes' via a pack, call our callback to handle
      onPack(lastUse->getOperandNumber());
      return;
    }

    Operation *nextOp = lastOp->getNextNode();
    if (!nextOp) {
      IVLOG(3, "  terminator");
      return;
    }

    IVLOG(3, "  next operation: " << debugString(*nextOp));
    OpBuilder builder(nextOp);
    builder.create<DeallocOp>(allocOp->getLoc(), lastUse->get());
    if (argNumber >= 0) {
      // Need to copy the initial argument
      auto scfFor = cast<scf::ForOp>(allocOp->getParentOp());
      builder.setInsertionPoint(scfFor);
      auto inits = scfFor.getIterOperands();
      auto argInit = inits[argNumber];
      MemRefType resType = argInit.getType().cast<MemRefType>();
      auto newBuf = builder.create<AllocOp>(builder.getUnknownLoc(), resType);
      auto forOp = builder.create<AffineParallelOp>(
          builder.getUnknownLoc(),
          /*resultTypes=*/ArrayRef<Type>{resType},
          /*reductions=*/ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign},
          /*ranges=*/resType.getShape());
      auto body = forOp.getBody();
      builder.setInsertionPointToStart(body);
      SmallVector<Value, 8> operandIdxs(resType.getRank());
      for (unsigned i = 0; i < resType.getRank(); i++) {
        operandIdxs[i] = body->getArgument(i);
      }
      auto loadRes = builder.create<pxa::PxaLoadOp>(builder.getUnknownLoc(),
                                                    argInit, operandIdxs);
      auto idMap = builder.getMultiDimIdentityMap(resType.getRank());
      auto storeRes = builder.create<pxa::PxaReduceOp>(
          builder.getUnknownLoc(), AtomicRMWKind::assign, loadRes, newBuf,
          idMap, builder.getBlock()->getArguments());
      builder.create<AffineYieldOp>(builder.getUnknownLoc(),
                                    ValueRange{storeRes.result()});
      // Replace the initial argument with the result of for loop
      scfFor.setOperand(scfFor.getNumControlOperands() + argNumber,
                        forOp.getResult(0));
    }
  }
};

std::unique_ptr<mlir::Pass> createDeallocPlacementPass() {
  return std::make_unique<DeallocPlacementPass>();
}

} // namespace pmlc::dialect::pxa
