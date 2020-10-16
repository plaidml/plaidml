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
  void makePlaidmlExec(mlir::OpBuilder &builder);
  void makePlaidmlFini(mlir::OpBuilder &builder);

  mlir::FuncOp entryFunc;
  mlir::FunctionType entryType;

  LLVMType llvmInt32Type;
  LLVMType llvmPtrTy;
  LLVMType llvmVoidTy;
};

void MakeEntrypointsPass::runOnOperation() {
  entryFunc = getNetworkEntry();
  if (!entryFunc) {
    return;
  }
  entryType = entryFunc.getType();

  initMembers();
  mlir::OpBuilder builder(getOperation().getBody()->getTerminator());
  makePlaidmlInit(builder);
  makePlaidmlExec(builder);
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

  for (auto func : getOperation().getOps<mlir::FuncOp>()) {
    if (func.isExternal() || func.getName() != networkMain) {
      continue;
    }
    if (entryFunc) {
      getOperation().emitError(
          "Expected only one entrypoint function definition");
      signalPassFailure();
      break;
    }
    entryFunc = func;
  }
  if (!entryFunc) {
    getOperation().emitError(
        "Expected a single entrypoint function definition");
    signalPassFailure();
  }

  return entryFunc;
}

void MakeEntrypointsPass::makePlaidmlInit(mlir::OpBuilder &builder) {
  // This method builds a trivial passthrough plaidml_init:
  //
  //   (Device*, memref...) plaidml_init(Device* device, memref...) {
  //     return device, memref...;
  //   }
  //
  // ... with one exception: if the existing entry function does not take an
  // initial pointer parameter, the device pointer is omitted from the return
  // values.

  auto entryInputTypes = entryType.getInputs();
  mlir::SmallVector<mlir::Type, 8> initInputTypes;
  bool addedDeviceArgument = false;
  if (!entryInputTypes.size() ||
      !entryInputTypes[0].isa<LLVM::LLVMPointerType>()) {
    addedDeviceArgument = true;
    initInputTypes.push_back(llvmPtrTy);
  }
  for (auto ty : entryInputTypes) {
    initInputTypes.push_back(ty);
  }
  auto func = builder.create<mlir::FuncOp>(
      getLoc(), util::kPlaidmlInit,
      builder.getFunctionType(initInputTypes, entryInputTypes));
  auto block = func.addEntryBlock();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);
  auto argsBegin = block->args_begin();
  if (addedDeviceArgument) {
    ++argsBegin;
  }
  builder.create<mlir::ReturnOp>(
      getLoc(), mlir::ArrayRef<mlir::Value>(argsBegin, block->args_end()));
}

void MakeEntrypointsPass::makePlaidmlExec(mlir::OpBuilder &builder) {
  // This method builds a trivial passthrough plaidml_exec:
  //
  //   void plaidml_exec(args...) {
  //     main(args...);
  //   }

  auto func = builder.create<mlir::FuncOp>(
      getLoc(), util::kPlaidmlExec,
      builder.getFunctionType(entryType.getInputs(), {}));
  auto block = func.addEntryBlock();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);
  builder.create<mlir::CallOp>(getLoc(), entryFunc, block->getArguments());
  builder.create<mlir::ReturnOp>(getLoc(), mlir::ArrayRef<mlir::Value>{});
}

void MakeEntrypointsPass::makePlaidmlFini(mlir::OpBuilder &builder) {
  // This method builds a trivial plaidml_fini:
  //
  //   void plaidml_fini(args...) {}

  auto func = builder.create<mlir::FuncOp>(
      getLoc(), util::kPlaidmlFini,
      builder.getFunctionType({entryType.getInputs()}, {}));
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
