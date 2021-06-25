// Copyright 2020, Intel Corporation

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/DebugStringHelper.h"
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
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace pxa = dialect::pxa;
namespace stdx = dialect::stdx;
namespace xsmm = dialect::xsmm;

namespace {

util::StrideArray getStrideArray(Value operand, AffineMap tileMap) {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto type = operand.getType().cast<MemRefType>();
  // TODO: check LogicalResult
  (void)getStridesAndOffset(type, strides, offset);
  AffineMap layoutMap =
      makeStridedLinearLayoutMap(strides, offset, operand.getContext());
  Optional<util::StrideArray> info =
      util::computeStrideArray(layoutMap.compose(tileMap));
  assert(info.hasValue() && "computeStrideArray must succeed");
  return *info;
}

SmallVector<util::StrideArray> getStrideArrays(ValueRange operands,
                                               ArrayAttr tileMapsAttr) {
  SmallVector<util::StrideArray> result;
  for (auto tuple : llvm::zip(operands, tileMapsAttr)) {
    Value operand;
    Attribute tileMapAttr;
    std::tie(operand, tileMapAttr) = tuple;
    AffineMap tileMap = tileMapAttr.cast<AffineMapAttr>().getValue();
    util::StrideArray strideArray = getStrideArray(operand, tileMap);
    result.emplace_back(strideArray);
  }
  return result;
}

struct IndicesCollector {
  Location loc;
  ConversionPatternRewriter &rewriter;
  SmallVector<Value> indices;

  IndicesCollector(Location loc, ConversionPatternRewriter &rewriter)
      : loc(loc), rewriter(rewriter) {}

  bool collect(ArrayAttr arrayAttr, ValueRange mapIndices) {
    unsigned prefix = 0;
    for (Attribute attr : arrayAttr) {
      AffineMap accessMap = attr.cast<AffineMapAttr>().getValue();
      size_t count = accessMap.getNumInputs();
      auto operands = mapIndices.slice(prefix, count);
      auto expanded = expandAffineMap(rewriter, loc, accessMap, operands);
      if (!expanded)
        return false;
      indices.append(expanded->begin(), expanded->end());
      prefix += count;
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

static SmallVector<int64_t> getIntegerValues(ArrayAttr attr) {
  SmallVector<int64_t> result;
  for (APInt value : attr.getAsValueRange<IntegerAttr>())
    result.push_back(value.getZExtValue());
  return result;
}

struct GemmLowering {
  pxa::PxaGenericOp op;
  ConversionPatternRewriter &rewriter;
  Location loc;
  pxa::PxaGenericOp::Adaptor adaptor;
  Type resultType;
  SmallVector<util::StrideArray> inputs;
  SmallVector<util::StrideArray> outputs;
  IndicesCollector collector;
  SmallVector<int64_t> tileSizes;
  ArrayAttr tileAttr;
  ArrayAttr leadingDimsAttr;
  ArrayRef<int64_t> batches;

  static constexpr size_t OPA_IDX = 0;
  static constexpr size_t OPB_IDX = 1;
  static constexpr size_t OPC_IDX = 0;

  static constexpr size_t LDA_IDX = 0;
  static constexpr size_t LDB_IDX = 2;
  static constexpr size_t LDC_IDX = 0;

  GemmLowering(pxa::PxaGenericOp op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter)
      : op(op), rewriter(rewriter), loc(op.getLoc()),
        adaptor(operands, op->getAttrDictionary()),
        resultType(rewriter.getI64Type()),
        inputs(getStrideArrays(adaptor.inputs(), op.inputTileMaps())),
        outputs(getStrideArrays(adaptor.outputs(), op.outputTileMaps())),
        collector(loc, rewriter), tileSizes(getIntegerValues(op.tile())),
        tileAttr(rewriter.getI64ArrayAttr(
            ArrayRef<int64_t>(tileSizes).take_front(3))),
        leadingDimsAttr(rewriter.getI64ArrayAttr(ArrayRef<int64_t>{
            inputs[OPA_IDX].strides[LDA_IDX],
            inputs[OPB_IDX].strides[LDB_IDX],
            outputs[OPC_IDX].strides[LDC_IDX],
        })),
        batches(ArrayRef<int64_t>(tileSizes).drop_front(3)) {
    IVLOG(3, "StrideArrays: A: " << inputs[OPA_IDX]
                                 << ", B: " << inputs[OPB_IDX]
                                 << ", C:" << outputs[OPC_IDX]);
  }

  LogicalResult performLowering() {
    if (!collector.collect(op.outputAccessMaps(), adaptor.outputIndices()) ||
        !collector.collect(op.inputAccessMaps(), adaptor.inputIndices()))
      return failure();

    if (batches.empty())
      return callSimpleGemm();

    if (batches.size() == 1)
      return callBatchedGemm(batches[0]);

    // There are additional reduction indices
    // call offset based batch reduce gemm
    return callBatchedOffsetsGemm();
  }

  LogicalResult callSimpleGemm() {
    auto dispatch = rewriter.create<xsmm::GemmDispatchF32Op>(
        loc, resultType,
        /*tile=*/tileAttr,
        /*leadingDims=*/leadingDimsAttr);

    rewriter.create<xsmm::GemmInvokeF32Op>(loc, ArrayRef<Type>(),
                                           /*ptr=*/dispatch,
                                           /*c=*/adaptor.outputs()[OPC_IDX],
                                           /*a=*/adaptor.inputs()[OPA_IDX],
                                           /*b=*/adaptor.inputs()[OPB_IDX],
                                           /*indices=*/collector.indices);

    op.replaceAllUsesWith(adaptor.outputs());
    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult callBatchedGemm(int64_t kBatches) {
    auto dispatch = rewriter.create<xsmm::BRGemmDispatchF32Op>(
        loc, resultType,
        /*tile=*/tileAttr,
        /*leadingDims=*/leadingDimsAttr);

    rewriter.create<xsmm::BRGemmInvokeF32Op>(
        loc, ArrayRef<Type>(),
        /*ptr=*/dispatch,
        /*c=*/adaptor.outputs()[OPC_IDX],
        /*a=*/adaptor.inputs()[OPA_IDX],
        /*b=*/adaptor.inputs()[OPB_IDX],
        /*numBatches=*/rewriter.getI64IntegerAttr(kBatches),
        /*indices=*/collector.indices);

    op.replaceAllUsesWith(adaptor.outputs());
    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult callBatchedOffsetsGemm() {
    // offsets for index k in matrix multiply and
    // additional reduction indices, stepSizes are the tileSizes for each
    // dimension
    SmallVector<int64_t> aStrides, bStrides, stepSizes;
    SmallVector<Type> inputTypes = getElementTypes(adaptor.inputs().getTypes());
    int64_t aElemSize = inputTypes[OPA_IDX].getIntOrFloatBitWidth() / 8;
    int64_t bElemSize = inputTypes[OPB_IDX].getIntOrFloatBitWidth() / 8;

    // Skip index m and n, start from k at index 2
    const util::StrideArray &aInfo = inputs[OPA_IDX];
    for (size_t i = 2; i < aInfo.strides.size(); i++)
      aStrides.emplace_back(aInfo.strides[i] * aElemSize);

    // Skip index m and n, start from k at index 2
    const util::StrideArray &bInfo = inputs[OPB_IDX];
    for (size_t i = 2; i < bInfo.strides.size(); i++)
      bStrides.emplace_back(bInfo.strides[i] * bElemSize);

    // Push the step size (tile size) for k
    int64_t kTile = (op.tile().getValue()[2]).cast<IntegerAttr>().getInt();
    stepSizes.emplace_back(kTile);

    // Rest of the reduction indices are unit step
    for (size_t i = 2; i < aInfo.strides.size(); i++)
      stepSizes.emplace_back(1);

    SmallVector<int64_t, 4> aOffsets, bOffsets;
    int64_t numSteps = 1;
    for (int64_t batchSize : batches)
      numSteps *= batchSize;

    // Computation of offset table
    computeBRGemmOffsets(batches, stepSizes, aStrides, bStrides, aOffsets,
                         bOffsets);

    auto dispatch = rewriter.create<xsmm::BRGemmOffsDispatchF32Op>(
        loc, resultType,
        /*tile=*/tileAttr,
        /*leadingDims=*/leadingDimsAttr);

    rewriter.create<xsmm::BRGemmOffsInvokeF32Op>(
        loc, ArrayRef<Type>(),
        /*ptr=*/dispatch,
        /*c=*/adaptor.outputs()[0],
        /*a=*/adaptor.inputs()[0],
        /*b=*/adaptor.inputs()[1],
        /*numBatches=*/rewriter.getI64IntegerAttr(numSteps),
        /*aOffsets=*/rewriter.getI64ArrayAttr(aOffsets),
        /*bOffsets=*/rewriter.getI64ArrayAttr(bOffsets),
        /*indices=*/collector.indices);

    op.replaceAllUsesWith(adaptor.outputs());
    rewriter.eraseOp(op);

    return success();
  }

  void computeBRGemmOffsets(ArrayRef<int64_t> numSteps,
                            ArrayRef<int64_t> stepSizes,
                            ArrayRef<int64_t> aStrides,
                            ArrayRef<int64_t> bStrides,
                            SmallVectorImpl<int64_t> &aOffsets,
                            SmallVectorImpl<int64_t> &bOffsets) {
    IVLOG(3, "computeBRGemmOffsets:")
    IVLOG(3, "  numSteps: " << numSteps);
    IVLOG(3, "  stepSizes: " << stepSizes);
    IVLOG(3, "  aStrides: " << aStrides);
    IVLOG(3, "  bStrides: " << bStrides);

    int64_t numBatches = 1;
    for (int64_t step : numSteps)
      numBatches *= step;
    IVLOG(3, "  numBatches: " << numBatches);

    aOffsets.resize(numBatches, 0);
    bOffsets.resize(numBatches, 0);

    assert((numSteps.size() == aStrides.size() &&
            numSteps.size() == bStrides.size()) &&
           "argument dimension mismatch for offset based BRGEMM");

    // memory stride of the current index within the offset array
    int64_t innerStride = 1;

    for (size_t i = 0; i < numSteps.size(); i++) {
      // memory stride for array a
      int64_t aStride = aStrides[i];
      // memory stride for array b
      int64_t bStride = bStrides[i];
      // the step size of this index
      int64_t stepSize = stepSizes[i];
      // the number of batches for this index
      int64_t nSteps = numSteps[i];

      for (int64_t k = 0; k < numBatches; k += (nSteps * innerStride)) {
        for (int64_t j = 0; j < nSteps; j++) {
          for (int64_t m = 0; m < innerStride; m++) {
            aOffsets[k + j * innerStride + m] += (j * stepSize * aStride);
            bOffsets[k + j * innerStride + m] += (j * stepSize * bStride);
          }
        }
      }

      // update inner memory stride for next index
      innerStride *= nSteps;
    }
  }
};

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

    GemmLowering lowering(op, operands, rewriter);
    return lowering.performLowering();
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
    SmallVector<util::StrideArray> inputs =
        getStrideArrays(adaptor.inputs(), op.inputTileMaps());
    SmallVector<util::StrideArray> outputs =
        getStrideArrays(adaptor.outputs(), op.outputTileMaps());
    IndicesCollector collector(loc, rewriter);
    if (!collector.collect(op.outputAccessMaps(), adaptor.outputIndices()) ||
        !collector.collect(op.inputAccessMaps(), adaptor.inputIndices()))
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
                                               /*ldi=*/inputs[0].strides[0],
                                               /*ldo=*/outputs[0].strides[0],
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
    static int aOffsetGlobalVarCount = 0;
    static int bOffsetGlobalVarCount = 0;

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

template <typename T>
static Optional<TppAccess> getTppAccess(MLIRContext *context, T op,
                                        Block *block,
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

  AffineValueMap innerValueMap;
  AffineValueMap outerValueMap = pxa::convertToValueMap(context, rap->outer);
  AffineValueMap accessValueMap(op.map(), op.idxs());
  AffineValueMap::difference(accessValueMap, outerValueMap, &innerValueMap);

  mapOperands.append(outerValueMap.getOperands().begin(),
                     outerValueMap.getOperands().end());

  return TppAccess{outerValueMap.getAffineMap(), innerValueMap.getAffineMap(),
                   strideOneIV};
}

struct TppReluPattern : public OpRewritePattern<AffineParallelOp> {
  using OpRewritePattern<AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineParallelOp op,
                                PatternRewriter &rewriter) const {
    using matchers::m_Any;

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
    SmallVector<Value> inputIndices, outputIndices;

    auto reduceOp = cast<pxa::PxaReduceOp>(reduce.getDefiningOp());
    Optional<TppAccess> output =
        getTppAccess(context, reduceOp, op->getBlock(), outputIndices);
    if (!output) {
      IVLOG(3, "TppReluPattern: Failed due to a non-strided access");
      return failure();
    }

    auto loadOp = cast<pxa::PxaLoadOp>(load.getDefiningOp());
    Optional<TppAccess> input =
        getTppAccess(context, loadOp, op->getBlock(), inputIndices);
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

    ArrayAttr reductions =
        rewriter.getI64ArrayAttr({static_cast<int64_t>(AtomicRMWKind::assign)});

    rewriter.replaceOpWithNewOp<pxa::PxaGenericOp>(
        op, reduce.getType(),
        /*inputs=*/ArrayRef<Value>{loadOp.memref()},
        /*outputs=*/ArrayRef<Value>{reduceOp.memref()},
        /*inputIndices=*/inputIndices,
        /*outputIndices=*/outputIndices,
        /*inputAccessMaps=*/inputAccessMaps,
        /*inputTileMaps=*/inputTileMaps,
        /*outputAccessMaps=*/outputAccessMaps,
        /*outputTileMaps=*/outputTileMaps,
        /*kernel=*/rewriter.getStringAttr("tpp_relu"),
        /*tile=*/tile,
        /*reductions=*/reductions);

    return success();
  }
};

struct TppPatternsPass : public TppPatternsBase<TppPatternsPass> {
  void runOnFunction() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.insert<TppReluPattern>(context);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void populatePXAGemmToXSMMConversionPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<GemmPxaGenericOpConversion>(context);
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
