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

  auto builder = mlir::OpBuilder{mainFunc};

  auto mainTy = mainFunc.getType();
  if (mainTy.getNumResults()) {
    mainFunc.emitError(
        llvm::formatv("Expected no outputs from \"{0}\"", networkMain));
    signalPassFailure();
    return;
  }

  // It's a little simpler to create a new function, add the loop and
  // terminator, clone the existing function's body into the loop, and delete
  // the original function -- it seems a little wasteful, but it's easier to
  // build the loop (otherwise, we'd need to build it without inserting it, move
  // the existing region into it, and then insert it), and the cloning makes it
  // easy to rewrite the block arguments correctly.
  auto newFunc =
      mlir::cast<mlir::FuncOp>(builder.insert(mainFunc.cloneWithoutRegions()));
  auto *newEntryBlock = newFunc.addEntryBlock();

  // Partition the function arguments: memrefs are passed to the loop (from
  // which they'll be passed into the cloned region), while non-memrefs are
  // added to the mapper (so that they'll be connected to the new function's
  // arguments)
  mlir::BlockAndValueMapping mapper;
  mlir::SmallVector<mlir::Value, 8> loopArgs;
  for (auto [mainArg, newArg] :
       llvm::zip(mainFunc.getArguments(), newFunc.getArguments())) {
    if (newArg.getType().isa<mlir::MemRefType>()) {
      loopArgs.emplace_back(newArg);
    } else {
      mapper.map(mainArg, newArg);
    }
  }

  // Create the new function's body.
  builder.setInsertionPointToStart(newEntryBlock);
  auto loopOp = builder.create<abi::LoopOp>(newFunc.getLoc(), loopArgs);
  builder.create<mlir::ReturnOp>(newFunc.getLoc());
  mainFunc.getRegion().cloneInto(&loopOp.getRegion(), mapper);

  // We no longer need the main function.
  mainFunc.erase();

  // Normally, functions end with std.return; since std.return expects its
  // parent to be std.func, we need to replace any cloned top-level std.return
  // operations with abi.done.
  for (auto &op : llvm::make_early_inc_range(loopOp.getOps())) {
    auto returnOp = mlir::dyn_cast<mlir::ReturnOp>(op);
    if (returnOp) {
      builder.setInsertionPoint(returnOp);
      builder.create<abi::DoneOp>(returnOp.getLoc());
      returnOp.erase();
    }
  }
}

} // namespace

std::unique_ptr<mlir::Pass> createAddABILoopPass() {
  return std::make_unique<AddABILoopPass>();
}

} // namespace pmlc::dialect::abi
