// Copyright 2020, Intel Corporation

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Module.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/xsmm/ir/ops.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace pxa = dialect::pxa;
namespace xsmm = dialect::xsmm;

namespace {

StrideArray getStrideArray(Value operand, AffineMap tileMap) {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto type = operand.getType().cast<MemRefType>();
  getStridesAndOffset(type, strides, offset);
  auto layoutMap =
      makeStridedLinearLayoutMap(strides, offset, operand.getContext());
  auto info = computeStrideArray(layoutMap.compose(tileMap));
  assert(info.hasValue() && "computeStrideArray must succeed");
  return *info;
}

struct AffineGemmOpConversion : public OpConversionPattern<pxa::AffineGemmOp> {
  using OpConversionPattern<pxa::AffineGemmOp>::OpConversionPattern;

  bool getIndices(pxa::AffineGemmOp op, ConversionPatternRewriter &rewriter,
                  pxa::AffineGemmOp::Adaptor &adaptor, AffineMap accessMap,
                  unsigned start, unsigned count,
                  SmallVectorImpl<Value> &into) const {
    auto operands = adaptor.mapOperands().slice(start, count);
    auto indices = expandAffineMap(rewriter, op.getLoc(), accessMap, operands);
    if (!indices)
      return false;
    into.append(indices->begin(), indices->end());
    return true;
  }

  LogicalResult
  matchAndRewrite(pxa::AffineGemmOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    pxa::AffineGemmOp::Adaptor transformed(operands);
    SmallVector<Value, 8> indices;
    auto aNumInputs = op.aAccessMap().getNumInputs();
    auto bNumInputs = op.bAccessMap().getNumInputs();
    auto cNumInputs = op.cAccessMap().getNumInputs();
    if (!getIndices(op, rewriter, transformed, op.cAccessMap(), 0, cNumInputs,
                    indices) ||
        !getIndices(op, rewriter, transformed, op.aAccessMap(), cNumInputs,
                    aNumInputs, indices) ||
        !getIndices(op, rewriter, transformed, op.bAccessMap(),
                    cNumInputs + aNumInputs, bNumInputs, indices))
      return failure();

    auto aInfo = getStrideArray(transformed.a(), op.aTileMap());
    auto bInfo = getStrideArray(transformed.b(), op.bTileMap());
    auto cInfo = getStrideArray(transformed.c(), op.cTileMap());
    auto leadingDimsAttr = rewriter.getI64ArrayAttr(ArrayRef<int64_t>{
        aInfo.strides[0], bInfo.strides[0], cInfo.strides[0]});

    auto dispatch = rewriter.create<xsmm::GemmDispatchOp>(
        op.getLoc(), rewriter.getI64Type(), op.tile(), leadingDimsAttr);

    rewriter.create<xsmm::GemmInvokeOp>(op.getLoc(), ArrayRef<Type>(), dispatch,
                                        transformed.c(), transformed.a(),
                                        transformed.b(), indices);

    op.replaceAllUsesWith(transformed.c());
    rewriter.eraseOp(op);

    return success();
  }
};

struct XSMMGemmDispatchLowering
    : public ConvertOpToLLVMPattern<xsmm::GemmDispatchOp> {
  using ConvertOpToLLVMPattern<xsmm::GemmDispatchOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto dispatchOp = cast<xsmm::GemmDispatchOp>(op);
    auto func = getOrInsertFunc(op, rewriter);

    auto int32Type = LLVM::LLVMType::getInt32Ty(&getDialect());
    auto int64Type = LLVM::LLVMType::getInt64Ty(&getDialect());
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
        op, int64Type, rewriter.getSymbolRefAttr(func), callOperands);
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
    auto int64Type = LLVM::LLVMType::getInt64Ty(&getDialect());
    return rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), kGemmDispatchF32,
        LLVM::LLVMType::getFunctionTy(int64Type,
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
    auto &module = getModule();
    auto invokeOp = cast<xsmm::GemmInvokeOp>(op);
    xsmm::GemmInvokeOp::Adaptor transformed(operands);
    auto aType = invokeOp.a().getType().cast<MemRefType>();
    auto bType = invokeOp.b().getType().cast<MemRefType>();
    auto cType = invokeOp.c().getType().cast<MemRefType>();

    auto aIndices =
        transformed.indices().slice(cType.getRank(), aType.getRank());
    auto aPtr = getDataPtr(op->getLoc(), aType, transformed.a(), aIndices,
                           rewriter, module);

    auto bIndices = transformed.indices().slice(
        cType.getRank() + aType.getRank(), bType.getRank());
    auto bPtr = getDataPtr(op->getLoc(), bType, transformed.b(), bIndices,
                           rewriter, module);

    auto cIndices = transformed.indices().slice(0, cType.getRank());
    auto cPtr = getDataPtr(op->getLoc(), cType, transformed.c(), cIndices,
                           rewriter, module);

    auto func = getOrInsertFunc(op, rewriter);
    rewriter.create<LLVM::CallOp>(
        op->getLoc(), ArrayRef<Type>(), rewriter.getSymbolRefAttr(func),
        ArrayRef<Value>{transformed.ptr(), aPtr, bPtr, cPtr});
    rewriter.eraseOp(op);

    return success();
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
    auto int64Type = LLVM::LLVMType::getInt64Ty(&getDialect());
    auto floatPtrType = floatType.getPointerTo();
    return rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), kGemmInvokeF32,
        LLVM::LLVMType::getFunctionTy(
            getVoidType(),
            ArrayRef<LLVM::LLVMType>{int64Type,     // funcPtr
                                     floatPtrType,  // a
                                     floatPtrType,  // b
                                     floatPtrType}, // c
            /*isVarArg=*/false));
  }
};

} // namespace

void populatePXAToAffineConversionPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *ctx) {
  patterns.insert<AffineGemmOpConversion>(ctx);
}

void populateXSMMToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          OwningRewritePatternList &patterns) {
  patterns.insert<XSMMGemmDispatchLowering, XSMMGemmInvokeLowering>(converter);
}

} // namespace pmlc::target::x86
