// Copyright 2020 Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/FormatVariadic.h"

#include "pmlc/dialect/abi/transforms/pass_detail.h"
#include "pmlc/util/ids.h"

namespace pmlc::dialect::abi {
namespace {

class AddABILoopPass final : public AddABILoopPassBase<AddABILoopPass> {
public:
  void runOnOperation() final;
};

void AddABILoopPass::runOnOperation() {
  auto moduleOp = getOperation();
  auto mainFunc =
      mlir::dyn_cast_or_null<mlir::FuncOp>(moduleOp.lookupSymbol(networkMain));
  if (!mainFunc) {
    moduleOp.emitError(llvm::formatv("Unable to resolve \"{0}\"", networkMain));
    signalPassFailure();
    return;
  }

  auto mainTy = mainFunc.getType();
  if (mainTy.getNumResults()) {
    mainFunc.emitError(
        llvm::formatv("Expected no outputs from \"{0}\"", networkMain));
    signalPassFailure();
    return;
  }

  // Create the loop.
  auto loc = mainFunc.getLoc();
  auto builder = mlir::OpBuilder{mainFunc};
  auto loopOp = builder.create<abi::LoopOp>(loc);

  // Initialize the loop's body by taking the main function's body.
  // N.B. We created the loop with the same argument types, so the entry
  // block will be compatible with the loop's signature.
  loopOp.bodyRegion().takeBody(mainFunc.getBody());

  // We no longer need the main function.
  mainFunc.erase();

  // A function's blocks can end with std.return; since std.return expects its
  // parent to be std.func, we need to replace any cloned top-level std.return
  // operations with abi.done.
  for (auto &op : llvm::make_early_inc_range(loopOp.bodyRegion().getOps())) {
    auto returnOp = mlir::dyn_cast<mlir::ReturnOp>(op);
    if (returnOp) {
      builder.setInsertionPoint(returnOp);
      builder.create<abi::DoneOp>(returnOp.getLoc());
      returnOp.erase();
    }
  }

  // Add an entry block for the init region.
  auto *initEntryBlock = builder.createBlock(&loopOp.initRegion());

  // Reorder the non-memref parameters in the body entry block.
  //
  // Memrefs will vary from invocation to invocation; they are intrinsically
  // parameters of the body entry block.  Non-memrefs are all going to be passed
  // via the network structure, so we reorder them in the body entry block,
  // putting them at the front of the argument list and plumbing them in
  // through the init block's parameters and the CreateNetworkOp terminator of
  // the init block.
  //
  // TODO: We should distinguish weight memrefs (constants) and non-weight
  //       memrefs (actual network inputs).  It might also be interesting to
  //       allow non-memrefs to vary from run to run.  Perhaps we could signify
  //       const-over-the-network-lifetime via attributes?
  auto *bodyEntryBlock = loopOp.bodyEntryBlock();
  mlir::Optional<unsigned> firstMemrefIdx;
  for (unsigned idx = 0; idx < bodyEntryBlock->getNumArguments(); ++idx) {
    auto arg = bodyEntryBlock->getArgument(idx);
    auto ty = arg.getType();
    if (ty.isa<mlir::MemRefType>()) {
      firstMemrefIdx = idx;
      continue;
    }
    if (firstMemrefIdx.hasValue()) {
      auto newArg =
          bodyEntryBlock->insertArgument(firstMemrefIdx.getValue(), ty);
      arg.replaceAllUsesWith(newArg);
      bodyEntryBlock->eraseArgument(idx + 1);
    }
    initEntryBlock->addArgument(ty);
  }

  // Finally, terminate the init block using a passthrough of the
  // init block's arguments.
  builder.create<abi::CreateNetworkOp>(loc, initEntryBlock->getArguments());

  // Finally, create the fini region's entry block.
  builder.createBlock(&loopOp.finiRegion());
  builder.create<abi::DoneOp>(loc);
}

} // namespace

std::unique_ptr<mlir::Pass> createAddABILoopPass() {
  return std::make_unique<AddABILoopPass>();
}

} // namespace pmlc::dialect::abi
