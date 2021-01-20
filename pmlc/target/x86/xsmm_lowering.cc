// Copyright 2020, Intel Corporation

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"

#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/xsmm/ir/ops.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/strides.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace pxa = dialect::pxa;
namespace xsmm = dialect::xsmm;

namespace {

util::StrideArray getStrideArray(Value operand, AffineMap tileMap) {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto type = operand.getType().cast<MemRefType>();
  getStridesAndOffset(type, strides, offset);
  auto layoutMap =
      makeStridedLinearLayoutMap(strides, offset, operand.getContext());
  auto info = util::computeStrideArray(layoutMap.compose(tileMap));
  assert(info.hasValue() && "computeStrideArray must succeed");
  return *info;
}

struct PxaGemmOpConversion : public OpConversionPattern<pxa::PxaGemmOp> {
  using OpConversionPattern<pxa::PxaGemmOp>::OpConversionPattern;

  bool getIndices(pxa::PxaGemmOp op, ConversionPatternRewriter &rewriter,
                  pxa::PxaGemmOp::Adaptor &adaptor, AffineMap accessMap,
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
  matchAndRewrite(pxa::PxaGemmOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    pxa::PxaGemmOp::Adaptor transformed(operands);
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

    int64_t numBatches = op.numBatches();

    if (numBatches == 1) {
      auto dispatch = rewriter.create<xsmm::GemmDispatchF32Op>(
          op.getLoc(), rewriter.getI64Type(), op.tile(), leadingDimsAttr);

      rewriter.create<xsmm::GemmInvokeF32Op>(
          op.getLoc(), ArrayRef<Type>(), dispatch, transformed.c(),
          transformed.a(), transformed.b(), indices);
    } else if (numBatches > 1) {
      auto dispatch = rewriter.create<xsmm::BRGemmDispatchF32Op>(
          op.getLoc(), rewriter.getI64Type(), op.tile(), leadingDimsAttr);

      rewriter.create<xsmm::BRGemmInvokeF32Op>(
          op.getLoc(), ArrayRef<Type>(), dispatch, transformed.c(),
          transformed.a(), transformed.b(), op.numBatches(), indices);
    } else {
      return failure();
    }

    op.replaceAllUsesWith(transformed.c());
    rewriter.eraseOp(op);

    return success();
  }
};

struct XSMMGemmDispatchF32Lowering
    : public ConvertOpToLLVMPattern<xsmm::GemmDispatchF32Op> {
  using ConvertOpToLLVMPattern<xsmm::GemmDispatchF32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(xsmm::GemmDispatchF32Op op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto func = getOrInsertFunc(op, rewriter);

    IntegerType int32Type = rewriter.getI32Type();
    IntegerType int64Type = rewriter.getI64Type();
    SmallVector<Value, 6> callOperands;

    // lda, ldb, ldc
    for (auto attr : op.tileld().getValue()) {
      callOperands.push_back(
          rewriter.create<LLVM::ConstantOp>(op->getLoc(), int32Type, attr));
    }

    // m, n, k
    for (auto attr : op.tile().getValue()) {
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
    IntegerType int32Type = rewriter.getI32Type();
    IntegerType int64Type = rewriter.getI64Type();
    return rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), kGemmDispatchF32,
        LLVM::LLVMFunctionType::get(int64Type,
                                    ArrayRef<Type>{int32Type,  // lda
                                                   int32Type,  // ldb
                                                   int32Type,  // ldc
                                                   int32Type,  // m
                                                   int32Type,  // n
                                                   int32Type}, // k
                                    /*isVarArg=*/false));
  }
};

struct XSMMGemmInvokeF32Lowering
    : public ConvertOpToLLVMPattern<xsmm::GemmInvokeF32Op> {
  using ConvertOpToLLVMPattern<xsmm::GemmInvokeF32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(xsmm::GemmInvokeF32Op op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    xsmm::GemmInvokeF32Op::Adaptor transformed(operands);
    auto aType = op.a().getType().cast<MemRefType>();
    auto bType = op.b().getType().cast<MemRefType>();
    auto cType = op.c().getType().cast<MemRefType>();

    auto aIndices =
        transformed.indices().slice(cType.getRank(), aType.getRank());
    auto aPtr = getStridedElementPtr(op->getLoc(), aType, transformed.a(),
                                     aIndices, rewriter);

    auto bIndices = transformed.indices().slice(
        cType.getRank() + aType.getRank(), bType.getRank());
    auto bPtr = getStridedElementPtr(op->getLoc(), bType, transformed.b(),
                                     bIndices, rewriter);

    auto cIndices = transformed.indices().slice(0, cType.getRank());
    auto cPtr = getStridedElementPtr(op->getLoc(), cType, transformed.c(),
                                     cIndices, rewriter);

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
    IntegerType int64Type = rewriter.getI64Type();
    auto floatPtrType = LLVM::LLVMPointerType::get(rewriter.getF32Type());
    return rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), kGemmInvokeF32,
        LLVM::LLVMFunctionType::get(getVoidType(),
                                    ArrayRef<Type>{int64Type,     // funcPtr
                                                   floatPtrType,  // a
                                                   floatPtrType,  // b
                                                   floatPtrType}, // c
                                    /*isVarArg=*/false));
  }
};

struct XSMMBRGemmDispatchF32Lowering
    : public ConvertOpToLLVMPattern<xsmm::BRGemmDispatchF32Op> {
  using ConvertOpToLLVMPattern<
      xsmm::BRGemmDispatchF32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(xsmm::BRGemmDispatchF32Op op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto func = getOrInsertFunc(op, rewriter);

    IntegerType int32Type = rewriter.getI32Type();
    IntegerType int64Type = rewriter.getI64Type();
    SmallVector<Value, 8> callOperands;

    // lda, ldb, ldc
    for (auto attr : op.tileld().getValue()) {
      callOperands.push_back(
          rewriter.create<LLVM::ConstantOp>(op->getLoc(), int32Type, attr));
    }

    // m, n, k
    for (auto attr : op.tile().getValue()) {
      callOperands.push_back(
          rewriter.create<LLVM::ConstantOp>(op->getLoc(), int32Type, attr));
    }

    /*
    // stride_a
    callOperands.push_back(
      rewriter.create<LLVM::ConstantOp>(op->getLoc(), int32Type,
    dispatchOp.StrideA()));
      */

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, int64Type, rewriter.getSymbolRefAttr(func), callOperands);
    return success();
  }

  LLVM::LLVMFuncOp getOrInsertFunc(Operation *op,
                                   ConversionPatternRewriter &rewriter) const {
    const char *kBRGemmDispatchF32 = "plaidml_rt_xsmm_brgemm_dispatch_f32";

    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(kBRGemmDispatchF32);
    if (func)
      return func;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    IntegerType int32Type = rewriter.getI32Type();
    IntegerType int64Type = rewriter.getI64Type();
    return rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), kBRGemmDispatchF32,
        LLVM::LLVMFunctionType::get(int64Type,
                                    ArrayRef<Type>{int32Type,  // lda
                                                   int32Type,  // ldb
                                                   int32Type,  // ldc
                                                   int32Type,  // m
                                                   int32Type,  // n
                                                   int32Type}, // k
                                    /*isVarArg=*/false));
  }
};

struct XSMMBRGemmInvokeF32Lowering
    : public ConvertOpToLLVMPattern<xsmm::BRGemmInvokeF32Op> {
  using ConvertOpToLLVMPattern<xsmm::BRGemmInvokeF32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(xsmm::BRGemmInvokeF32Op op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    xsmm::BRGemmInvokeF32Op::Adaptor transformed(operands);
    auto aType = op.a().getType().cast<MemRefType>();
    auto bType = op.b().getType().cast<MemRefType>();
    auto cType = op.c().getType().cast<MemRefType>();

    auto aIndices =
        transformed.indices().slice(cType.getRank(), aType.getRank());
    auto aPtr = getStridedElementPtr(op->getLoc(), aType, transformed.a(),
                                     aIndices, rewriter);

    auto bIndices = transformed.indices().slice(
        cType.getRank() + aType.getRank(), bType.getRank());
    auto bPtr = getStridedElementPtr(op->getLoc(), bType, transformed.b(),
                                     bIndices, rewriter);

    auto cIndices = transformed.indices().slice(0, cType.getRank());
    auto cPtr = getStridedElementPtr(op->getLoc(), cType, transformed.c(),
                                     cIndices, rewriter);

    IntegerType int64Type = rewriter.getI64Type();
    auto numBatches = rewriter.getI64IntegerAttr(op.numBatches());
    auto numBatchesValue =
        rewriter.create<LLVM::ConstantOp>(op->getLoc(), int64Type, numBatches);

    auto func = getOrInsertFunc(op, rewriter);
    rewriter.create<LLVM::CallOp>(
        op->getLoc(), ArrayRef<Type>(), rewriter.getSymbolRefAttr(func),
        ArrayRef<Value>{transformed.ptr(), aPtr, bPtr, cPtr, numBatchesValue});
    rewriter.eraseOp(op);

    return success();
  }

  LLVM::LLVMFuncOp getOrInsertFunc(Operation *op,
                                   ConversionPatternRewriter &rewriter) const {
    const char *kBRGemmInvokeF32 = "plaidml_rt_xsmm_brgemm_invoke_f32";

    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(kBRGemmInvokeF32);
    if (func)
      return func;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    IntegerType int64Type = rewriter.getI64Type();
    auto floatPtrType = LLVM::LLVMPointerType::get(rewriter.getF32Type());
    return rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), kBRGemmInvokeF32,
        LLVM::LLVMFunctionType::get(getVoidType(),
                                    ArrayRef<Type>{int64Type,    // funcPtr
                                                   floatPtrType, // a
                                                   floatPtrType, // b
                                                   floatPtrType, // c
                                                   int64Type},   // numBatches
                                    /*isVarArg=*/false));
  }
};

} // namespace

void populatePXAGemmToXSMMConversionPatterns(OwningRewritePatternList &patterns,
                                             MLIRContext *ctx) {
  patterns.insert<PxaGemmOpConversion>(ctx);
}

void populateXSMMToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          OwningRewritePatternList &patterns) {
  patterns.insert<XSMMGemmDispatchF32Lowering, XSMMGemmInvokeF32Lowering,
                  XSMMBRGemmDispatchF32Lowering, XSMMBRGemmInvokeF32Lowering>(
      converter);
}
} // namespace pmlc::target::x86
