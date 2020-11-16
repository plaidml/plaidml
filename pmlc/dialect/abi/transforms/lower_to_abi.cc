// Copyright 2020 Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/FormatVariadic.h"

#include "pmlc/dialect/abi/transforms/pass_detail.h"
#include "pmlc/util/ids.h"

using LLVMType = mlir::LLVM::LLVMType;

namespace pmlc::dialect::abi {
namespace {

class LowerToABIPass final : public LowerToABIPassBase<LowerToABIPass> {
public:
  void runOnOperation() final;
};

void LowerToABIPass::runOnOperation() {
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
  loopOp.bodyRegion().takeBody(mainFunc.getBody());

  // A function's blocks can end with std.return; since std.return expects its
  // parent to be std.func, we need to replace any cloned top-level std.return
  // operations with abi.terminator.
  for (auto &op : llvm::make_early_inc_range(loopOp.bodyRegion().getOps())) {
    auto returnOp = mlir::dyn_cast<mlir::ReturnOp>(op);
    if (returnOp) {
      builder.setInsertionPoint(returnOp);
      builder.create<abi::TerminatorOp>(returnOp.getLoc());
      returnOp.erase();
    }
  }

  // Add an entry block for the init region.
  auto *initEntryBlock = builder.createBlock(&loopOp.initRegion());

  // Non-constant arguments will vary from invocation to invocation; they are
  // intrinsically parameters of the body entry block.  Constant arguments do
  // not vary; we pass them to the initialization block, which can then pass
  // them on to the body as needed.
  //
  // At this point, all of the arguments should be memrefs, except for a
  // possible initial device argument -- which, if present, is always passed
  // to the initialization block.
  auto *bodyEntryBlock = loopOp.getBodyEntryBlock();
  if (bodyEntryBlock->getNumArguments() == 0 ||
      bodyEntryBlock->getArgument(0)
          .getType()
          .isa<mlir::MemRefType, mlir::UnrankedMemRefType>()) {
    // The body entry block started with a memref (or didn't have any
    // arguments --we should always have at least *one* argument, but we handle
    // the no-argument case for completeness).  So, create a fake device
    // parameter.
    initEntryBlock->addArgument(
        LLVMType::getInt8Ty(&getContext()).getPointerTo());
  }

  mlir::Identifier constAttrId = builder.getIdentifier("tile.const");
  mlir::SmallVector<mlir::Value, 8> networkArgs;
  mlir::SmallVector<mlir::Type, 8> networkTypes;
  for (auto arg : bodyEntryBlock->getArguments()) {
    bool isMemRef =
        arg.getType().isa<mlir::MemRefType, mlir::UnrankedMemRefType>();
    auto argIdx = arg.getArgNumber();
    if (argIdx && !isMemRef) {
      mainFunc.emitError(
          "Expected only memrefs after an optional device parameter");
      signalPassFailure();
      return;
    }
    auto constAttr =
        mainFunc.getArgAttrOfType<mlir::IntegerAttr>(argIdx, constAttrId);
    if (constAttr || (!argIdx && !isMemRef)) {
      auto ty = arg.getType();
      auto newArg = bodyEntryBlock->insertArgument(networkArgs.size(), ty);
      arg.replaceAllUsesWith(newArg);
      bodyEntryBlock->eraseArgument(argIdx + 1);
      initEntryBlock->addArgument(ty);
      networkArgs.emplace_back(
          initEntryBlock->getArgument(initEntryBlock->getNumArguments() - 1));
      networkTypes.emplace_back(ty);
    }
  }

  // Terminate the init block using a passthrough of the
  // init block's arguments.
  builder.create<abi::YieldOp>(loc, networkArgs);

  // Add the fini region.
  auto *finiEntryBlock = builder.createBlock(&loopOp.finiRegion());
  finiEntryBlock->addArguments(networkTypes);
  builder.create<abi::TerminatorOp>(loc);

  // We no longer need the main function.
  mainFunc.erase();
}

} // namespace

std::unique_ptr<mlir::Pass> createLowerToABIPass() {
  return std::make_unique<LowerToABIPass>();
}

} // namespace pmlc::dialect::abi
