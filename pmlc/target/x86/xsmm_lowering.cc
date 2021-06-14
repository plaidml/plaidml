// Copyright 2020, Intel Corporation

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "libxsmm.h" // NOLINT [build/include_subdir]

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/ir/matchers.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/xsmm/ir/ops.h"
#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/matchers.h"
#include "pmlc/util/strides.h"
#include "pmlc/util/tags.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace pxa = dialect::pxa;
namespace stdx = dialect::stdx;
namespace xsmm = dialect::xsmm;
using matchers::m_Any;

namespace {

int aOffsetGlobalVarCount = 0;
int bOffsetGlobalVarCount = 0;

util::StrideArray getStrideArray(Value operand, AffineMap tileMap) {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto type = operand.getType().cast<MemRefType>();
  // TODO: check LogicalResult
  (void)getStridesAndOffset(type, strides, offset);
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

  void computeBRGemmOffsets(const SmallVector<int64_t, 4> &numSteps,
                            const SmallVector<int64_t, 4> &stepSizes,
                            const SmallVector<int64_t, 4> &aStrides,
                            const SmallVector<int64_t, 4> &bStrides,
                            SmallVector<int64_t, 4> &aOffsetsArray,
                            SmallVector<int64_t, 4> &bOffsetsArray) const {
    int numBatches = 1;
    for (size_t i = 0; i < numSteps.size(); i++) {
      numBatches *= numSteps[i];
    }

    aOffsetsArray = SmallVector<int64_t, 8>(numBatches, 0);
    bOffsetsArray = SmallVector<int64_t, 8>(numBatches, 0);

    assert((numSteps.size() == aStrides.size() &&
            numSteps.size() == bStrides.size()) &&
           "argument dimension mismatch for offset based BRGEMM");

    IVLOG(3, "numBatches in computeBRGEMM: " << numBatches);

    // variable to record the memory stride of the current index
    // within the offset array
    size_t innerStride = 1;

    for (size_t i = 0; i < numSteps.size(); i++) {
      // memory stride for array a
      int64_t aStride = aStrides[i];
      // memory stride for array b
      int64_t bStride = bStrides[i];
      // the iteration range of this index
      int64_t indexRange = stepSizes[i] * numSteps[i];
      // the number of batches for this index
      int64_t nSteps = numSteps[i];

      IVLOG(3, "aStride in computeBRGEMM: " << aStride);
      IVLOG(3, "bStride in computeBRGEMM: " << bStride);
      IVLOG(3, "indexRange in computeBRGEMM: " << indexRange);
      IVLOG(3, "numSteps in computeBRGEMM: " << nSteps);

      for (size_t k = 0; k < (size_t)numBatches;
           k += ((size_t)nSteps * innerStride)) {
        for (size_t j = 0; j < (size_t)nSteps; j++) {
          for (size_t l = 0; l < innerStride; l++) {
            aOffsetsArray[k + j * innerStride + l] +=
                (j * (indexRange / nSteps) * aStride);
            bOffsetsArray[k + j * innerStride + l] +=
                (j * (indexRange / nSteps) * bStride);
          }
        }
      }
      // update inner memory stride for next index
      innerStride *= (size_t)nSteps;
    }
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

    auto numBatches = op.numBatches();
    SmallVector<int64_t, 4> numBatchesArr;
    for (auto i : numBatches.getValue()) {
      numBatchesArr.emplace_back(i.cast<IntegerAttr>().getInt());
    }
    // If numbatches only consists of 'k' index call xsmm gemm or xsmm brgemm
    if (numBatchesArr.size() == 1) {
      int numBatches = numBatchesArr[0];
      // If value of numbatches is 1 call xsmm gemm
      if (numBatches == 1) {
        auto dispatch = rewriter.create<xsmm::GemmDispatchF32Op>(
            op.getLoc(), rewriter.getI64Type(), op.tile(), leadingDimsAttr);

        rewriter.create<xsmm::GemmInvokeF32Op>(
            op.getLoc(), ArrayRef<Type>(), dispatch, transformed.c(),
            transformed.a(), transformed.b(), indices);

        // Else call batch reduce gemm when number of batches is greater than 1.
      } else if (numBatches > 1) {
        auto numBatchesAttr = rewriter.getI64IntegerAttr(numBatches);
        auto dispatch = rewriter.create<xsmm::BRGemmDispatchF32Op>(
            op.getLoc(), rewriter.getI64Type(), op.tile(), leadingDimsAttr);

        rewriter.create<xsmm::BRGemmInvokeF32Op>(
            op.getLoc(), ArrayRef<Type>(), dispatch, transformed.c(),
            transformed.a(), transformed.b(), numBatchesAttr, indices);
      }
    } else if (numBatchesArr.size() > 1) {
      // There are additional reduction indices
      // call offset based batch reduce gemm

      // offsets for index k in matrix multiply and
      // additional reduction indices, stepSizes are the tilesizes for each
      // dimension
      SmallVector<int64_t, 4> aStrides, bStrides, stepSizes;
      int64_t aElemSize = transformed.a()
                              .getType()
                              .cast<MemRefType>()
                              .getElementType()
                              .getIntOrFloatBitWidth() /
                          8;
      int64_t bElemSize = transformed.b()
                              .getType()
                              .cast<MemRefType>()
                              .getElementType()
                              .getIntOrFloatBitWidth() /
                          8;
      // Skip index i, start from k at index 1
      for (size_t i = 1; i < aInfo.strides.size(); i++) {
        aStrides.emplace_back(aInfo.strides[i] * aElemSize);
      }

      bStrides.emplace_back(bInfo.strides[0] * bElemSize);
      // Skip 1st index which is j.
      for (size_t i = 2; i < bInfo.strides.size(); i++) {
        bStrides.emplace_back(bInfo.strides[i] * bElemSize);
      }

      // Push the step size (tile size) for k
      int64_t kTile = (op.tile().getValue()[2]).cast<IntegerAttr>().getInt();
      stepSizes.emplace_back(kTile);

      // Rest of the reduction indices are unit step
      for (size_t i = 2; i < aInfo.strides.size(); i++)
        stepSizes.emplace_back(1);

      SmallVector<int64_t, 4> aOffsets, bOffsets;
      int64_t numSteps = 1;
      for (size_t i = 0; i < numBatchesArr.size(); i++) {
        numSteps *= numBatchesArr[i];
      }

      // Computation of offset table
      computeBRGemmOffsets(numBatchesArr, stepSizes, aStrides, bStrides,
                           aOffsets, bOffsets);
      auto dispatch = rewriter.create<xsmm::BRGemmOffsDispatchF32Op>(
          op.getLoc(), rewriter.getI64Type(), op.tile(), leadingDimsAttr);

      rewriter.create<xsmm::BRGemmOffsInvokeF32Op>(
          op.getLoc(), ArrayRef<Type>(), dispatch, transformed.c(),
          transformed.a(), transformed.b(),
          rewriter.getI64IntegerAttr(numSteps),
          rewriter.getI64ArrayAttr(ArrayRef<int64_t>(aOffsets)),
          rewriter.getI64ArrayAttr(ArrayRef<int64_t>(bOffsets)), indices);
    } else {
      return failure();
    }

    op.replaceAllUsesWith(transformed.c());
    rewriter.eraseOp(op);

    return success();
  }
};

struct DispatchHelper {
  SmallVector<util::StrideArray> strides;
  SmallVector<int64_t> leadingDims;

  DispatchHelper(ValueRange operands, ArrayAttr tileMapsAttr) {
    for (auto tuple : llvm::zip(operands, tileMapsAttr)) {
      Value operand;
      Attribute tileMapAttr;
      std::tie(operand, tileMapAttr) = tuple;
      AffineMap tileMap = tileMapAttr.cast<AffineMapAttr>().getValue();
      util::StrideArray strideArray = getStrideArray(operand, tileMap);
      strides.push_back(strideArray);
      leadingDims.push_back(strideArray.strides[0]);
    }
  }
};

struct IndicesCollector {
  pxa::PxaGenericOp op;
  ConversionPatternRewriter &rewriter;
  pxa::PxaGenericOp::Adaptor &adaptor;
  SmallVector<Value> indices;
  unsigned prefix = 0;

  IndicesCollector(pxa::PxaGenericOp op, ConversionPatternRewriter &rewriter,
                   pxa::PxaGenericOp::Adaptor &adaptor)
      : op(op), rewriter(rewriter), adaptor(adaptor) {}

  bool collect(ArrayAttr arrayAttr) {
    for (Attribute attr : arrayAttr) {
      AffineMap accessMap = attr.cast<AffineMapAttr>().getValue();
      auto operands = adaptor.indices().slice(prefix, accessMap.getNumInputs());
      auto expanded =
          expandAffineMap(rewriter, op.getLoc(), accessMap, operands);
      if (!expanded)
        return false;
      indices.append(expanded->begin(), expanded->end());
      prefix += accessMap.getNumInputs();
    }
    return true;
  }
};

static SmallVector<Type> getElementTypes(TypeRange types) {
  SmallVector<Type> ret;
  for (Type type : types) {
    ret.push_back(type.cast<MemRefType>().getElementType());
  }
  return ret;
}

struct GemmPxaGenericOpConversion
    : public OpConversionPattern<pxa::PxaGenericOp> {
  using OpConversionPattern<pxa::PxaGenericOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(pxa::PxaGenericOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.kernel() != "tpp_gemm")
      return failure();
    if (op.outputs().size() != 1 || op.inputs().size() != 2)
      return failure();

    Location loc = op.getLoc();
    pxa::PxaGenericOp::Adaptor adaptor(operands, op->getAttrDictionary());
    Type resultType = rewriter.getI64Type();
    DispatchHelper inputs(adaptor.inputs(), op.input_tile_maps());
    DispatchHelper outputs(adaptor.outputs(), op.output_tile_maps());
    IndicesCollector collector(op, rewriter, adaptor);
    if (!collector.collect(op.output_access_maps()) ||
        !collector.collect(op.input_access_maps()))
      return failure();

    ArrayAttr leadingDimsAttr = rewriter.getI64ArrayAttr(ArrayRef<int64_t>{
        inputs.leadingDims[0], inputs.leadingDims[1], outputs.leadingDims[0]});
    auto dispatch = rewriter.create<xsmm::GemmDispatchF32Op>(
        loc, resultType,
        /*tile=*/op.tile(),
        /*leadingDims=*/leadingDimsAttr);

    rewriter.create<xsmm::GemmInvokeF32Op>(loc, ArrayRef<Type>(),
                                           /*ptr=*/dispatch,
                                           /*c=*/adaptor.outputs()[0],
                                           /*a=*/adaptor.inputs()[0],
                                           /*b=*/adaptor.inputs()[1],
                                           /*indices=*/collector.indices);

    op.replaceAllUsesWith(adaptor.outputs());
    rewriter.eraseOp(op);

    return success();
  }
};

struct UnaryPxaGenericOpConversion
    : public OpConversionPattern<pxa::PxaGenericOp> {
  StringRef kernelName;
  xsmm::UnaryKind kind;

  UnaryPxaGenericOpConversion(MLIRContext *context, StringRef kernelName,
                              xsmm::UnaryKind kind)
      : OpConversionPattern(context), kernelName(kernelName), kind(kind) {}

  LogicalResult
  matchAndRewrite(pxa::PxaGenericOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.kernel() != kernelName)
      return failure();
    if (op.outputs().size() != 1 || op.inputs().size() != 1)
      return failure();

    Location loc = op.getLoc();
    pxa::PxaGenericOp::Adaptor adaptor(operands, op->getAttrDictionary());
    Type resultType = rewriter.getI64Type();
    DispatchHelper inputs(adaptor.inputs(), op.input_tile_maps());
    DispatchHelper outputs(adaptor.outputs(), op.output_tile_maps());
    IndicesCollector collector(op, rewriter, adaptor);
    if (!collector.collect(op.output_access_maps()) ||
        !collector.collect(op.input_access_maps()))
      return failure();

    SmallVector<Type> inputTypes = getElementTypes(op.inputs().getTypes());
    SmallVector<Type> outputTypes = getElementTypes(op.outputs().getTypes());
    Type computeType = outputTypes[0]; // just use the output type for now
    FunctionType funcType = rewriter.getFunctionType(inputTypes, outputTypes);

    auto dispatchOp =
        rewriter.create<xsmm::UnaryDispatchOp>(loc, resultType,
                                               /*kind=*/kind,
                                               /*compute_type=*/computeType,
                                               /*tile=*/op.tile(),
                                               /*ldi=*/inputs.leadingDims[0],
                                               /*ldo=*/outputs.leadingDims[0],
                                               /*func_type=*/funcType);

    rewriter.create<xsmm::UnaryInvokeOp>(loc, ArrayRef<Type>(),
                                         /*ptr=*/dispatchOp,
                                         /*output=*/adaptor.outputs()[0],
                                         /*input=*/adaptor.inputs()[0],
                                         /*indices=*/collector.indices);

    op.replaceAllUsesWith(adaptor.outputs());
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
    for (auto attr : op.leadingDims().getValue()) {
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
    for (auto attr : op.leadingDims().getValue()) {
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

struct XSMMBRGemmOffsDispatchF32Lowering
    : public ConvertOpToLLVMPattern<xsmm::BRGemmOffsDispatchF32Op> {
  using ConvertOpToLLVMPattern<
      xsmm::BRGemmOffsDispatchF32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(xsmm::BRGemmOffsDispatchF32Op op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto func = getOrInsertFunc(op, rewriter);

    IntegerType int32Type = rewriter.getI32Type();
    IntegerType int64Type = rewriter.getI64Type();
    SmallVector<Value, 8> callOperands;

    // lda, ldb, ldc
    for (auto attr : op.leadingDims().getValue()) {
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
    const char *kBRGemmOffsDispatchF32 =
        "plaidml_rt_xsmm_brgemm_offs_dispatch_f32";

    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(kBRGemmOffsDispatchF32);
    if (func)
      return func;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    IntegerType int32Type = rewriter.getI32Type();
    IntegerType int64Type = rewriter.getI64Type();
    return rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), kBRGemmOffsDispatchF32,
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

struct XSMMBRGemmOffsInvokeF32Lowering
    : public ConvertOpToLLVMPattern<xsmm::BRGemmOffsInvokeF32Op> {
  using ConvertOpToLLVMPattern<
      xsmm::BRGemmOffsInvokeF32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(xsmm::BRGemmOffsInvokeF32Op op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    xsmm::BRGemmOffsInvokeF32Op::Adaptor transformed(operands);
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

    auto module = op->getParentOfType<ModuleOp>();
    OpBuilder builder(module.getBodyRegion());

    auto aOffsetType = RankedTensorType::get({numBatches.getInt()}, int64Type);
    auto bOffsetType = RankedTensorType::get({numBatches.getInt()}, int64Type);
    LLVM::GlobalOp aOffsets;
    LLVM::GlobalOp bOffsets;
    std::string aGlobalVar =
        "brgemm_aoffsets" + std::to_string(aOffsetGlobalVarCount++);
    std::string bGlobalVar =
        "brgemm_boffsets" + std::to_string(bOffsetGlobalVarCount++);

    aOffsets = builder.create<LLVM::GlobalOp>(
        builder.getUnknownLoc(),
        LLVM::LLVMArrayType::get(int64Type, numBatches.getInt()),
        /*isConstant=*/true, LLVM::Linkage::Internal, StringRef(aGlobalVar),
        DenseElementsAttr::get(aOffsetType, op.aOffsets().getValue()));
    bOffsets = builder.create<LLVM::GlobalOp>(
        builder.getUnknownLoc(),
        LLVM::LLVMArrayType::get(int64Type, numBatches.getInt()),
        /*isConstant=*/true, LLVM::Linkage::Internal, StringRef(bGlobalVar),
        DenseElementsAttr::get(bOffsetType, op.bOffsets().getValue()));

    auto longPtrType = LLVM::LLVMPointerType::get(rewriter.getI64Type());

    auto aOffsetsBase =
        rewriter.create<LLVM::AddressOfOp>(op->getLoc(), aOffsets);

    auto bOffsetsBase =
        rewriter.create<LLVM::AddressOfOp>(op->getLoc(), bOffsets);

    SmallVector<Value, 4> aOffsetOperands = {aOffsetsBase};
    aOffsetOperands.insert(aOffsetOperands.end(), aOffsetType.getRank() + 1,
                           createIndexConstant(rewriter, op->getLoc(), 0));

    SmallVector<Value, 4> bOffsetOperands = {bOffsetsBase};
    bOffsetOperands.insert(bOffsetOperands.end(), bOffsetType.getRank() + 1,
                           createIndexConstant(rewriter, op->getLoc(), 0));

    auto aOffsetsPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), longPtrType,
                                                    aOffsetOperands);

    auto bOffsetsPtr = rewriter.create<LLVM::GEPOp>(op->getLoc(), longPtrType,
                                                    bOffsetOperands);

    auto func = getOrInsertFunc(op, rewriter);
    rewriter.create<LLVM::CallOp>(
        op->getLoc(), ArrayRef<Type>(), rewriter.getSymbolRefAttr(func),
        ArrayRef<Value>{transformed.ptr(), aPtr, bPtr, cPtr, numBatchesValue,
                        aOffsetsPtr, bOffsetsPtr});
    rewriter.eraseOp(op);

    return success();
  }

  LLVM::LLVMFuncOp getOrInsertFunc(Operation *op,
                                   ConversionPatternRewriter &rewriter) const {
    const char *kBRGemmInvokeF32 = "plaidml_rt_xsmm_brgemm_offs_invoke_f32";
    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(kBRGemmInvokeF32);
    if (func)
      return func;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    IntegerType int64Type = rewriter.getI64Type();
    auto floatPtrType = LLVM::LLVMPointerType::get(rewriter.getF32Type());
    auto longPtrType = LLVM::LLVMPointerType::get(rewriter.getI64Type());
    return rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), kBRGemmInvokeF32,
        LLVM::LLVMFunctionType::get(getVoidType(),
                                    ArrayRef<Type>{int64Type,    // funcPtr
                                                   floatPtrType, // a
                                                   floatPtrType, // b
                                                   floatPtrType, // c
                                                   int64Type,    // numBatches
                                                   longPtrType,  // aOffsetsPtr
                                                   longPtrType}, // bOffsetsPtr
                                    /*isVarArg=*/false));
  }
};

static libxsmm_datatype convertDataType(Type type) {
  if (type.isF16())
    return LIBXSMM_DATATYPE_F16;
  if (type.isBF16())
    return LIBXSMM_DATATYPE_BF16;
  if (type.isF32())
    return LIBXSMM_DATATYPE_F32;
  if (type.isF64())
    return LIBXSMM_DATATYPE_F64;
  if (auto intType = type.dyn_cast<IntegerType>()) {
    switch (intType.getWidth()) {
    case 8:
      return LIBXSMM_DATATYPE_I8;
    case 16:
      return LIBXSMM_DATATYPE_I16;
    case 32:
      return LIBXSMM_DATATYPE_I32;
    case 64:
      return LIBXSMM_DATATYPE_I64;
    }
  }
  return LIBXSMM_DATATYPE_UNSUPPORTED;
}

struct XSMMUnaryDispatchLowering
    : public ConvertOpToLLVMPattern<xsmm::UnaryDispatchOp> {
  using ConvertOpToLLVMPattern<xsmm::UnaryDispatchOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(xsmm::UnaryDispatchOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    LLVM::LLVMFuncOp func = getOrInsertFunc(op, rewriter);

    IntegerType int32Type = rewriter.getI32Type();
    IntegerType int64Type = rewriter.getI64Type();
    SmallVector<Value, 8> callOperands;

    // m, n
    for (Attribute attr : op.tile().getValue()) {
      callOperands.push_back(
          rewriter.create<LLVM::ConstantOp>(loc, int32Type, attr));
    }

    // ldi
    callOperands.push_back(
        rewriter.create<LLVM::ConstantOp>(loc, int32Type, op.ldiAttr()));

    // ldo
    callOperands.push_back(
        rewriter.create<LLVM::ConstantOp>(loc, int32Type, op.ldoAttr()));

    FunctionType funcType = op.func_type();
    Type inputType = funcType.getInput(0);
    Type outputType = funcType.getResult(0);

    // in_type
    callOperands.push_back(rewriter.create<LLVM::ConstantOp>(
        loc, int32Type,
        rewriter.getI32IntegerAttr(convertDataType(inputType))));

    // compute_type
    callOperands.push_back(rewriter.create<LLVM::ConstantOp>(
        loc, int32Type,
        rewriter.getI32IntegerAttr(convertDataType(op.compute_type()))));

    // out_type
    callOperands.push_back(rewriter.create<LLVM::ConstantOp>(
        loc, int32Type,
        rewriter.getI32IntegerAttr(convertDataType(outputType))));

    // type
    callOperands.push_back(rewriter.create<LLVM::ConstantOp>(
        loc, int32Type,
        rewriter.getI32IntegerAttr(
            static_cast<int32_t>(op.kindAttr().getValue()))));

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, int64Type, rewriter.getSymbolRefAttr(func), callOperands);

    return success();
  }

  LLVM::LLVMFuncOp getOrInsertFunc(Operation *op,
                                   ConversionPatternRewriter &rewriter) const {
    const char *symbolName = "plaidml_rt_xsmm_unary_dispatch";

    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(symbolName);
    if (func)
      return func;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    IntegerType int32Type = rewriter.getI32Type();
    IntegerType int64Type = rewriter.getI64Type();
    return rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), symbolName,
        LLVM::LLVMFunctionType::get(int64Type,
                                    ArrayRef<Type>{
                                        int32Type, // m
                                        int32Type, // n
                                        int32Type, // ldi
                                        int32Type, // ldo
                                        int32Type, // in_type
                                        int32Type, // compute_type
                                        int32Type, // out_type
                                        int32Type, // type
                                    },
                                    /*isVarArg=*/false));
  }
};

struct XSMMUnaryInvokeLowering
    : public ConvertOpToLLVMPattern<xsmm::UnaryInvokeOp> {
  using ConvertOpToLLVMPattern<xsmm::UnaryInvokeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(xsmm::UnaryInvokeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    xsmm::UnaryInvokeOp::Adaptor transformed(operands);
    auto inputType = op.input().getType().cast<MemRefType>();
    auto outputType = op.output().getType().cast<MemRefType>();
    Type voidPtrType = getVoidPtrType();

    auto inputIndices =
        transformed.indices().slice(outputType.getRank(), inputType.getRank());
    Value inputPtr = getStridedElementPtr(loc, inputType, transformed.input(),
                                          inputIndices, rewriter);
    Value inputVoidPtr =
        rewriter.create<LLVM::BitcastOp>(loc, voidPtrType, inputPtr);

    auto outputIndices = transformed.indices().slice(0, outputType.getRank());
    Value outputPtr = getStridedElementPtr(
        loc, outputType, transformed.output(), outputIndices, rewriter);
    Value outputVoidPtr =
        rewriter.create<LLVM::BitcastOp>(loc, voidPtrType, outputPtr);

    LLVM::LLVMFuncOp func = getOrInsertFunc(op, rewriter);
    rewriter.create<LLVM::CallOp>(loc, ArrayRef<Type>(),
                                  rewriter.getSymbolRefAttr(func),
                                  ArrayRef<Value>{
                                      transformed.ptr(),
                                      inputVoidPtr,
                                      outputVoidPtr,
                                  });
    rewriter.eraseOp(op);

    return success();
  }

  LLVM::LLVMFuncOp getOrInsertFunc(Operation *op,
                                   ConversionPatternRewriter &rewriter) const {
    const char *symbolName = "plaidml_rt_xsmm_unary_invoke";

    auto module = op->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(symbolName);
    if (func)
      return func;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    IntegerType int64Type = rewriter.getI64Type();
    Type voidPtrType = getVoidPtrType();
    return rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), symbolName,
        LLVM::LLVMFunctionType::get(getVoidType(),
                                    ArrayRef<Type>{
                                        int64Type,   // ptr
                                        voidPtrType, // input
                                        voidPtrType, // output
                                    },
                                    /*isVarArg=*/false));
  }
};

struct TppAccess {
  AffineMap accessMap;
  AffineMap tileMap;
  BlockArgument strideOneIV;
};

static AffineMap makeTileMap(MLIRContext *context, AffineValueMap valueMap,
                             ArrayRef<BlockArgument> idxs) {
  SmallVector<AffineExpr, 8> exprs;
  for (auto value : valueMap.getOperands()) {
    bool found = false;
    for (size_t i = 0; i < idxs.size(); i++) {
      if (value == idxs[i]) {
        exprs.push_back(getAffineDimExpr(i, context));
        found = true;
      }
    }
    if (!found) {
      exprs.push_back(getAffineConstantExpr(0, context));
    }
  }
  auto toIdxs = AffineMap::get(idxs.size(), 0, exprs, context);
  return valueMap.getAffineMap().compose(toIdxs);
}

template <typename T>
static Optional<TppAccess> getTppAccess(T op, Block *block,
                                        ArrayRef<BlockArgument> tileIdxs,
                                        SmallVectorImpl<Value> &mapOperands) {
  Optional<pxa::RelativeAccessPattern> rap =
      pxa::computeRelativeAccess(op, block);
  if (!rap)
    return None;

  Optional<pxa::StrideInfo> flatInner = rap->flatInner();
  if (!flatInner)
    return None;

  BlockArgument strideOneIV;
  for (auto &info : flatInner->strides) {
    if (info.second == 1) {
      strideOneIV = info.first;
      break;
    }
  }

  if (!strideOneIV)
    return None;

  MLIRContext *context = op->getContext();
  AffineValueMap outerValueMap = pxa::convertToValueMap(context, rap->outer);
  AffineValueMap innerValueMap = pxa::convertToValueMap(context, rap->inner);
  AffineMap tileMap = makeTileMap(context, innerValueMap, tileIdxs);

  mapOperands.append(outerValueMap.getOperands().begin(),
                     outerValueMap.getOperands().end());

  return TppAccess{outerValueMap.getAffineMap(), tileMap, strideOneIV};
}

struct TppGemmPattern : public OpRewritePattern<AffineParallelOp> {
  using OpRewritePattern<AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineParallelOp op,
                                PatternRewriter &rewriter) const {
    SmallVector<int64_t> stencil;
    if (!getIntegerArrayTag(op, "stencil", stencil))
      return failure();

    IVLOG(1, "stencil: " << stencil);

    Value load1, load2, reduce;
    Operation *yield = op.getBody()->getTerminator();
    if (!matchPattern(
            yield,
            m_Op<AffineYieldOp>(m_Capture(
                &reduce,
                m_PxaReduceOp(
                    AtomicRMWKind::addf,
                    m_Op<MulFOp>(m_Capture(&load1, m_Op<pxa::PxaLoadOp>()),
                                 m_Capture(&load2, m_Op<pxa::PxaLoadOp>())),
                    m_Any())))) &&
        !matchPattern(
            yield,
            m_Op<AffineYieldOp>(m_Capture(
                &reduce,
                m_PxaReduceOp(
                    AtomicRMWKind::addi,
                    m_Op<MulIOp>(m_Capture(&load1, m_Op<pxa::PxaLoadOp>()),
                                 m_Capture(&load2, m_Op<pxa::PxaLoadOp>())),
                    m_Any()))))) {
      IVLOG(3, "TppGemmPattern: Pattern not found");
      return failure();
    }

    ArrayRef<BlockArgument> ivs = op.getIVs();
    SmallVector<Value> indices;

    auto reduceOp = cast<pxa::PxaReduceOp>(reduce.getDefiningOp());
    Optional<TppAccess> output =
        getTppAccess(reduceOp, op->getBlock(), {ivs[0], ivs[1]}, indices);
    if (!output) {
      IVLOG(3, "TppGemmPattern: Failed due to a non-strided access");
      return failure();
    }

    auto loadOp1 = cast<pxa::PxaLoadOp>(load1.getDefiningOp());
    Optional<TppAccess> input1 =
        getTppAccess(loadOp1, op->getBlock(), {ivs[0], ivs[2]}, indices);
    if (!input1) {
      IVLOG(3, "TppGemmPattern: Failed due to a non-strided access");
      return failure();
    }

    auto loadOp2 = cast<pxa::PxaLoadOp>(load2.getDefiningOp());
    Optional<TppAccess> input2 =
        getTppAccess(loadOp2, op->getBlock(), {ivs[2], ivs[1]}, indices);
    if (!input2) {
      IVLOG(3, "TppReluPattern: Failed due to a non-strided access");
      return failure();
    }

    // Optional<SmallVector<int64_t, 8>> ranges = op.getConstantRanges();
    // if (!ranges)
    //   return failure();

    ArrayAttr tile = rewriter.getI64ArrayAttr(stencil);

    ArrayAttr outputAccessMaps =
        rewriter.getAffineMapArrayAttr({output->accessMap});
    ArrayAttr outputTileMaps =
        rewriter.getAffineMapArrayAttr({output->tileMap});

    ArrayAttr inputAccessMaps =
        rewriter.getAffineMapArrayAttr({input1->accessMap, input2->accessMap});
    ArrayAttr inputTileMaps =
        rewriter.getAffineMapArrayAttr({input1->tileMap, input2->tileMap});

    rewriter.replaceOpWithNewOp<pxa::PxaGenericOp>(
        op, reduce.getType(),
        /*inputs=*/ArrayRef<Value>{loadOp1.memref(), loadOp2.memref()},
        /*outputs=*/ArrayRef<Value>{reduceOp.memref()},
        /*indices=*/indices,
        /*input_access_maps=*/inputAccessMaps,
        /*input_tile_maps=*/inputTileMaps,
        /*output_access_maps=*/outputAccessMaps,
        /*output_tile_maps=*/outputTileMaps,
        /*kernel=*/rewriter.getStringAttr("tpp_gemm"),
        /*tile=*/tile);

    return success();
  }
};

struct TppReluPattern : public OpRewritePattern<AffineParallelOp> {
  using OpRewritePattern<AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineParallelOp op,
                                PatternRewriter &rewriter) const {
    if (op.getIVs().size() != 2)
      return failure();

    Value load, reduce;
    auto pattern = m_Op<AffineYieldOp>(m_Capture(
        &reduce, pxa::m_PxaReduceOp(AtomicRMWKind::assign,
                                    m_Op<stdx::ReluOp>(m_Capture(
                                        &load, m_Op<pxa::PxaLoadOp>())),
                                    m_Any())));
    Operation *yield = op.getBody()->getTerminator();
    if (!matchPattern(yield, pattern)) {
      IVLOG(3, "TppReluPattern: Pattern not found");
      return failure();
    }

    if (!load.getType().isF32() ||
        !reduce.getType().cast<MemRefType>().getElementType().isF32())
      return failure();

    MLIRContext *context = rewriter.getContext();
    SmallVector<Value> indices;

    auto reduceOp = cast<pxa::PxaReduceOp>(reduce.getDefiningOp());
    Optional<TppAccess> output =
        getTppAccess(reduceOp, op->getBlock(), op.getIVs(), indices);
    if (!output) {
      IVLOG(3, "TppReluPattern: Failed due to a non-strided access");
      return failure();
    }

    auto loadOp = cast<pxa::PxaLoadOp>(load.getDefiningOp());
    Optional<TppAccess> input =
        getTppAccess(loadOp, op->getBlock(), op.getIVs(), indices);
    if (!input) {
      IVLOG(3, "TppReluPattern: Failed due to a non-strided access");
      return failure();
    }

    if (output->strideOneIV != input->strideOneIV) {
      IVLOG(3, "TppReluPattern: could not find compatible stride 1 IV");
      return failure();
    }

    Optional<SmallVector<int64_t, 8>> ranges = op.getConstantRanges();
    if (!ranges)
      return failure();

    ArrayAttr tile = rewriter.getI64ArrayAttr(*ranges);

    ArrayAttr outputAccessMaps =
        rewriter.getAffineMapArrayAttr({output->accessMap});
    ArrayAttr outputTileMaps =
        rewriter.getAffineMapArrayAttr({output->tileMap});

    ArrayAttr inputAccessMaps =
        rewriter.getAffineMapArrayAttr({input->accessMap});
    ArrayAttr inputTileMaps = rewriter.getAffineMapArrayAttr({input->tileMap});

    rewriter.replaceOpWithNewOp<pxa::PxaGenericOp>(
        op, reduce.getType(),
        /*inputs=*/ArrayRef<Value>{loadOp.memref()},
        /*outputs=*/ArrayRef<Value>{reduceOp.memref()},
        /*indices=*/indices,
        /*input_access_maps=*/inputAccessMaps,
        /*input_tile_maps=*/inputTileMaps,
        /*output_access_maps=*/outputAccessMaps,
        /*output_tile_maps=*/outputTileMaps,
        /*kernel=*/rewriter.getStringAttr("tpp_relu"),
        /*tile=*/tile);

    return success();
  }
};

struct TppPatternsPass : public TppPatternsBase<TppPatternsPass> {
  void runOnFunction() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.insert<TppGemmPattern, //
                    TppReluPattern  //
                    >(context);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void populatePXAGemmToXSMMConversionPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<GemmPxaGenericOpConversion, PxaGemmOpConversion>(context);
  patterns.insert<UnaryPxaGenericOpConversion>(context, "tpp_relu",
                                               xsmm::UnaryKind::RELU);
}

void populateXSMMToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns) {
  patterns.insert<XSMMGemmDispatchF32Lowering,       //
                  XSMMGemmInvokeF32Lowering,         //
                  XSMMBRGemmDispatchF32Lowering,     //
                  XSMMBRGemmInvokeF32Lowering,       //
                  XSMMBRGemmOffsDispatchF32Lowering, //
                  XSMMBRGemmOffsInvokeF32Lowering,   //
                  XSMMUnaryDispatchLowering,         //
                  XSMMUnaryInvokeLowering            //
                  >(converter);
}

std::unique_ptr<mlir::Pass> createTppPatternsPass() {
  return std::make_unique<TppPatternsPass>();
}

} // namespace pmlc::target::x86
