// Copyright 2020, Intel Corporation

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Module.h"

#include "pmlc/dialect/xsmm/ir/ops.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace xsmm = dialect::xsmm;

namespace {

struct XSMMGemmDispatchLowering
    : public ConvertOpToLLVMPattern<xsmm::GemmDispatchOp> {
  using ConvertOpToLLVMPattern<xsmm::GemmDispatchOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto dispatchOp = cast<xsmm::GemmDispatchOp>(op);
    auto func = getOrInsertFunc(op, rewriter);

    auto int32Type = LLVM::LLVMType::getInt32Ty(&getDialect());
    SmallVector<Value, 6> callOperands;

    // lda, ldb, ldc
    for (auto attr : dispatchOp.tileld().getValue()) {
      callOperands.push_back(
          rewriter.create<LLVM::ConstantOp>(op->getLoc(), int32Type, attr));
    }

    // m, n, k
    for (auto attr : dispatchOp.tile().getValue()) {
      callOperands.push_back(
          rewriter.create<LLVM::ConstantOp>(op->getLoc(), int32Type, attr));
    }

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, getVoidPtrType(), rewriter.getSymbolRefAttr(func), callOperands);
    return success();
  }

  LLVM::LLVMFuncOp getOrInsertFunc(Operation *op,
                                   ConversionPatternRewriter &rewriter) const {
    const char *kGemmDispatchF32 = "plaidml_rt_xsmm_gemm_dispatch_f32";

    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(kGemmDispatchF32);
    if (func)
      return func;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto int32Type = LLVM::LLVMType::getInt32Ty(&getDialect());
    return rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), kGemmDispatchF32,
        LLVM::LLVMType::getFunctionTy(getVoidPtrType(),
                                      ArrayRef<LLVM::LLVMType>{int32Type, // lda
                                                               int32Type, // ldb
                                                               int32Type, // ldc
                                                               int32Type, // m
                                                               int32Type, // n
                                                               int32Type}, // k
                                      /*isVarArg=*/false));
  }
};

struct XSMMGemmInvokeLowering
    : public ConvertOpToLLVMPattern<xsmm::GemmInvokeOp> {
  using ConvertOpToLLVMPattern<xsmm::GemmInvokeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto invokeOp = cast<xsmm::GemmInvokeOp>(op);
    xsmm::GemmInvokeOp::Adaptor transformed(operands);

    auto aIndices =
        transformed.mapOperands().slice(invokeOp.cAccessMap().getNumInputs(),
                                        invokeOp.aAccessMap().getNumInputs());
    auto aPtr = getOperandPtr(invokeOp, rewriter, invokeOp.a().getType(),
                              transformed.a(), invokeOp.aAccessMap(), aIndices);
    if (!aPtr)
      return failure();

    auto bIndices = transformed.mapOperands().slice(
        invokeOp.cAccessMap().getNumInputs() +
            invokeOp.aAccessMap().getNumInputs(),
        invokeOp.bAccessMap().getNumInputs());
    auto bPtr = getOperandPtr(invokeOp, rewriter, invokeOp.b().getType(),
                              transformed.b(), invokeOp.bAccessMap(), bIndices);
    if (!bPtr)
      return failure();

    auto cIndices = transformed.mapOperands().slice(
        0, invokeOp.cAccessMap().getNumInputs());
    auto cPtr = getOperandPtr(invokeOp, rewriter, invokeOp.c().getType(),
                              transformed.c(), invokeOp.cAccessMap(), cIndices);
    if (!cPtr)
      return failure();

    auto func = getOrInsertFunc(op, rewriter);
    rewriter.create<LLVM::CallOp>(op->getLoc(), ArrayRef<Type>(),
                                  rewriter.getSymbolRefAttr(func),
                                  ArrayRef<Value>{transformed.ptr(), // funcPtr
                                                  *aPtr,             // a
                                                  *bPtr,             // b
                                                  *cPtr});           // c
    rewriter.eraseOp(op);

    return success();
  }

  Optional<Value> getOperandPtr(xsmm::GemmInvokeOp op,
                                ConversionPatternRewriter &rewriter, Type type,
                                Value memref, AffineMap accessMap,
                                ValueRange indices) const {
    auto operands = expandAffineMap(rewriter, op.getLoc(), accessMap, indices);
    if (!operands)
      return llvm::None;
    return getDataPtr(op.getLoc(), type.cast<MemRefType>(), memref, *operands,
                      rewriter, getModule());
  }

  LLVM::LLVMFuncOp getOrInsertFunc(Operation *op,
                                   ConversionPatternRewriter &rewriter) const {
    const char *kGemmInvokeF32 = "plaidml_rt_xsmm_gemm_invoke_f32";

    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(kGemmInvokeF32);
    if (func)
      return func;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto floatType = LLVM::LLVMType::getFloatTy(&getDialect());
    auto floatPtrType = floatType.getPointerTo();
    return rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), kGemmInvokeF32,
        LLVM::LLVMType::getFunctionTy(
            getVoidType(),
            ArrayRef<LLVM::LLVMType>{getVoidPtrType(), // funcPtr
                                     floatPtrType,     // a
                                     floatPtrType,     // b
                                     floatPtrType},    // c
            /*isVarArg=*/false));
  }
};

} // namespace

void populateXSMMToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          OwningRewritePatternList &patterns,
                                          const LowerToLLVMOptions &options) {
  patterns.insert<XSMMGemmDispatchLowering, XSMMGemmInvokeLowering>(converter,
                                                                    options);
}

} // namespace pmlc::target::x86
