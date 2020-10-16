// Copyright 2020 Intel Corporation

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"

#include "pmlc/transforms/pass_detail.h"
#include "pmlc/util/ids.h"

namespace LLVM = mlir::LLVM;
using LLVM::LLVMFuncOp;
using LLVM::LLVMFunctionType;
using LLVM::LLVMType;

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

namespace pmlc::transforms {
namespace {

class BoxingPass final : public BoxingPassBase<BoxingPass> {
public:
  void runOnOperation() final;
};

void BoxingPass::runOnOperation() {
  auto builder = mlir::OpBuilder::atBlockBegin(getOperation().getBody());

  // Ensure that we have malloc().
  auto moduleOp = getOperation();
  auto mallocFunc = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("malloc");
  if (!mallocFunc) {
    mallocFunc = builder.create<LLVM::LLVMFuncOp>(
        builder.getUnknownLoc(), "malloc",
        LLVMType::getFunctionTy(
            LLVMType::getInt8PtrTy(&getContext()),
            {LLVMType::getInt64Ty(
                &getContext()) /* TODO: Use the correct size_t bitwidth */},
            /*isVarArg=*/false));
  }

  // Process the existing functions.
  for (auto func : moduleOp.getOps<LLVMFuncOp>()) {
    if (func.isExternal() || !func.isPublic()) {
      continue;
    }

    // Create the wrapper function declaration.
    builder.setInsertionPoint(func);
    auto funcTy = func.getType().dyn_cast<LLVMFunctionType>();
    auto inTy = LLVMType::getStructTy(&getContext(), funcTy.getParams());
    auto outTy = funcTy.getReturnType();
    if (outTy.isVoidTy()) {
      // TODO: Verify this assumption.
      // Lowering to LLVM IR fails on !llvm.ptr<void>, so we use !llvm.ptr<i8>
      // instead.
      outTy = LLVMType::getInt8Ty(&getContext());
    }
    auto wrapTy = LLVMType::getFunctionTy(
        outTy.getPointerTo(), {inTy.getPointerTo()}, /*isVarArg=*/false);
    auto wrapFunc = builder.create<LLVMFuncOp>(
        func.getLoc(), "_mlir_wrapper_" + func.getName().str(), wrapTy);
    auto wrapBody = wrapFunc.addEntryBlock();
    builder.setInsertionPointToStart(wrapBody);

    // Unwrap the incoming pointer-to-struct-of-arguments, and build up the
    // function call to the real function.
    auto inPtr = wrapBody->getArgument(0);
    auto inVal = builder.create<LLVM::LoadOp>(func.getLoc(), inPtr);
    mlir::SmallVector<mlir::Value, 8> callArgs;
    int argIdx = 0;
    for (auto ty : funcTy.getParams()) {
      auto arg = builder.create<LLVM::ExtractValueOp>(
          func.getLoc(), ty, inVal, builder.getIndexArrayAttr({argIdx++}));
      callArgs.emplace_back(arg);
    }
    auto call = builder.create<LLVM::CallOp>(func.getLoc(), func, callArgs);

    // Build up the return value.
    mlir::Value retValue;
    if (funcTy.getReturnType().isVoidTy()) {
      // If the wrapped function returns void, we just return a null
      // pointer.
      retValue =
          builder.create<LLVM::NullOp>(func.getLoc(), outTy.getPointerTo());
    } else {
      // If the wrapped function returns void, we just return a null pointer.
      // Otherwise, we need to find the size of the value to return,
      // malloc that much memory, fill it in, and return it.

      // Unfortunately, LLVMIR lacks a sizeof() operation.  Fortunately, it has
      // GetElementPtr.
      auto nullPtr =
          builder.create<LLVM::NullOp>(func.getLoc(), outTy.getPointerTo());
      auto nextEltIdx = builder.create<LLVM::ConstantOp>(
          func.getLoc(), LLVMType::getInt32Ty(&getContext()),
          builder.getIndexAttr(1));
      auto nextElt = builder.create<LLVM::GEPOp>(
          func.getLoc(), outTy.getPointerTo(), nullPtr,
          mlir::ArrayRef<mlir::Value>{nextEltIdx});
      auto size = builder.create<LLVM::PtrToIntOp>(
          func.getLoc(), LLVMType::getInt64Ty(&getContext()), nextElt);
      auto buffer = builder
                        .create<LLVM::CallOp>(func.getLoc(), mallocFunc,
                                              mlir::ArrayRef<mlir::Value>{size})
                        .getResult(0);
      retValue = builder.create<LLVM::BitcastOp>(func.getLoc(),
                                                 outTy.getPointerTo(), buffer);
      builder.create<LLVM::StoreOp>(func.getLoc(), call.getResult(0), retValue);
    }
    builder.create<LLVM::ReturnOp>(func.getLoc(),
                                   mlir::ArrayRef<mlir::Value>{retValue});
  }
}

} // namespace

std::unique_ptr<mlir::Pass> createBoxingPass() {
  return std::make_unique<BoxingPass>();
}
} // namespace pmlc::transforms
