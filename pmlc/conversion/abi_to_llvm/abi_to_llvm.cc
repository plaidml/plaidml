// Copyright 2020, Intel Corporation

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/conversion/abi_to_llvm/passes.h"
#include "pmlc/dialect/abi/ir/dialect.h"
#include "pmlc/util/ids.h"

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

namespace abi = pmlc::dialect::abi;
namespace LLVM = mlir::LLVM;
using LLVMType = LLVM::LLVMType;

namespace pmlc::conversion::abi_to_llvm {

static abi::LoopOp findLoopToLower(LLVM::LLVMFuncOp funcOp) {
  auto moduleOp = funcOp.getParentOfType<mlir::ModuleOp>();
  auto existingInit = moduleOp.lookupSymbol(pmlc::util::kPlaidmlInit);
  if (existingInit) {
    return abi::LoopOp{};
  }

  abi::LoopOp loopOp;
  for (auto childOp : funcOp.getOps<abi::LoopOp>()) {
    if (loopOp) {
      return abi::LoopOp{}; // Two loop ops => failure
    }
    loopOp = childOp;
  }

  return loopOp;
}

namespace {

struct LoopLowering : public mlir::ConvertOpToLLVMPattern<LLVM::LLVMFuncOp> {
  using mlir::ConvertOpToLLVMPattern<LLVM::LLVMFuncOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto funcOp = mlir::cast<LLVM::LLVMFuncOp>(op);
    auto loopOp = findLoopToLower(funcOp);
    if (!loopOp) {
      return mlir::failure();
    }

    llvm::errs() << "Converting fn: " << funcOp << "\n";

    auto ctx = rewriter.getContext();
    auto networkTy = LLVMType::createStructTy(ctx, mlir::StringRef{"Network"});

    auto initFunc = rewriter.create<LLVM::LLVMFuncOp>(
        funcOp.getLoc(), pmlc::util::kPlaidmlInit,
        LLVMType::getFunctionTy(
            networkTy.getPointerTo(),
            funcOp.getType().cast<LLVM::LLVMFunctionType>().getParams(),
            /*isVarArg=*/false));

    auto execFunc = rewriter.create<LLVM::LLVMFuncOp>(
        funcOp.getLoc(), pmlc::util::kPlaidmlExec,
        LLVMType::getFunctionTy(getVoidType(), {networkTy.getPointerTo()},
                                /*isVarArg=*/false));

    auto finiFunc = rewriter.create<LLVM::LLVMFuncOp>(
        funcOp.getLoc(), pmlc::util::kPlaidmlFini,
        LLVMType::getFunctionTy(getVoidType(), {networkTy.getPointerTo()},
                                /*isVarArg=*/false));

    rewriter.updateRootInPlace(funcOp, [&] {});

    return mlir::success();
  }
};

} // namespace

void populateABIToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter,
    mlir::OwningRewritePatternList &patterns) {
  patterns.insert<LoopLowering>(converter);
}

void addLoopLegality(mlir::ConversionTarget &target) {
  target.addDynamicallyLegalOp<LLVM::LLVMFuncOp>([](LLVM::LLVMFuncOp op) {
    if (findLoopToLower(op)) {
      // We have a loop to lower, so this op is not legal.
      return false;
    }
    // With no loop to lower, there's nothing for us to do.
    return true;
  });
}

} // namespace pmlc::conversion::abi_to_llvm
