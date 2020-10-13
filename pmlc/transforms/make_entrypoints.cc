// Copyright 2020 Intel Corporation

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"

#include "pmlc/transforms/pass_detail.h"
#include "pmlc/util/ids.h"

namespace LLVM = mlir::LLVM;
using LLVMType = LLVM::LLVMType;

namespace pmlc::transforms {
namespace {

class MakeEntrypointsPass final
    : public MakeEntrypointsPassBase<MakeEntrypointsPass> {
public:
  void runOnOperation() final;

private:
  mlir::Location getLoc() { return getOperation().getLoc(); }
  void initMembers();
  mlir::FuncOp getNetworkEntry();

  void makePlaidmlInit(mlir::OpBuilder &builder);
  void makePlaidmlExecute(mlir::OpBuilder &builder, mlir::FuncOp entryFunc);
  void makePlaidmlFini(mlir::OpBuilder &builder);

  LLVMType llvmInt32Type;
  LLVMType llvmPtrTy;
  LLVMType llvmVoidTy;
};

void MakeEntrypointsPass::runOnOperation() {
  mlir::FuncOp entryFunc = getNetworkEntry();
  if (!entryFunc) {
    return;
  }

  initMembers();
  mlir::OpBuilder builder(getOperation().getBody()->getTerminator());
  makePlaidmlInit(builder);
  makePlaidmlExecute(builder, entryFunc);
  makePlaidmlFini(builder);
}

void MakeEntrypointsPass::initMembers() {
  auto ctx = &getContext();
  llvmInt32Type = LLVMType::getInt32Ty(ctx);
  llvmPtrTy = LLVMType::getInt8PtrTy(ctx);
  llvmVoidTy = LLVMType::getVoidTy(ctx);
}

mlir::FuncOp MakeEntrypointsPass::getNetworkEntry() {
  mlir::FuncOp entryFunc;

  getOperation().walk([&](mlir::FuncOp func) {
    if (func.isExternal() || func.getName() != networkMain) {
      return mlir::WalkResult::advance();
    }
    if (entryFunc) {
      getOperation().emitError(
          "Expected only one entrypoint function definition");
      signalPassFailure();
      return mlir::WalkResult::interrupt();
    }
    entryFunc = func;
    return mlir::WalkResult::advance();
  });
  if (!entryFunc) {
    getOperation().emitError(
        "Expected a single entrypoint function definition");
    signalPassFailure();
  }

  return entryFunc;
}

void MakeEntrypointsPass::makePlaidmlInit(mlir::OpBuilder &builder) {
  // This method builds:
  //
  //   Device* plaidml_init(Device* device) {
  //     return device;
  //   }
  //
  // This is a rather trivial function; the plan is to expand it later.
  auto func = builder.create<mlir::FuncOp>(
      getLoc(), kPlaidmlInit,
      builder.getFunctionType({llvmPtrTy}, {llvmPtrTy}));
  auto block = func.addEntryBlock();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);
  builder.create<mlir::ReturnOp>(
      getLoc(), mlir::ArrayRef<mlir::Value>{block->getArgument(0)});
}

void MakeEntrypointsPass::makePlaidmlExecute(mlir::OpBuilder &builder,
                                             mlir::FuncOp entryFunc) {
  // This method builds:
  //
  //   void plaidml_execute(Device* device, memref...) {
  //     // N.B. main() may not take a Device parameter.
  //     main(device, memref...);
  //   }
  //
  // This is a rather trivial function; the plan is to expand it later.

  auto entryType = entryFunc.getType();
  auto entryInputTypes = entryType.getInputs();
  mlir::SmallVector<mlir::Type, 8> inputTypes;
  bool addedDeviceArgument = false;
  if (!entryInputTypes.size() ||
      !entryInputTypes[0].isa<LLVM::LLVMPointerType>()) {
    addedDeviceArgument = true;
    inputTypes.push_back(llvmPtrTy);
  }
  for (auto ty : entryInputTypes) {
    inputTypes.push_back(ty);
  }
  auto func = builder.create<mlir::FuncOp>(
      getLoc(), kPlaidmlExecute, builder.getFunctionType(inputTypes, {}));
  auto block = func.addEntryBlock();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);
  auto args = block->getArguments();
  mlir::SmallVector<mlir::Value, 8> callArgs;
  auto firstIt = args.begin();
  if (addedDeviceArgument) {
    ++firstIt;
  }
  std::copy(firstIt, args.end(), std::back_inserter(callArgs));
  builder.create<mlir::CallOp>(getLoc(), entryFunc, callArgs);
  builder.create<mlir::ReturnOp>(getLoc(), mlir::ArrayRef<mlir::Value>{});
}

void MakeEntrypointsPass::makePlaidmlFini(mlir::OpBuilder &builder) {
  // This method builds:
  //
  //   void teardown(Device* device) {
  //   }
  //
  // This is a rather trivial function; the plan is to expand it later.
  auto func = builder.create<mlir::FuncOp>(
      getLoc(), kPlaidmlFini, builder.getFunctionType({llvmPtrTy}, {}));
  auto block = func.addEntryBlock();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);
  builder.create<mlir::ReturnOp>(getLoc(), mlir::ArrayRef<mlir::Value>{});
}

} // namespace

std::unique_ptr<mlir::Pass> createMakeEntrypointsPass() {
  return std::make_unique<MakeEntrypointsPass>();
}

} // namespace pmlc::transforms
