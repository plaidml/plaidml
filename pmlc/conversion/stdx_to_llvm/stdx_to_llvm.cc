// Copyright 2020, Intel Corporation

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/conversion/stdx_to_llvm/pass_detail.h"
#include "pmlc/conversion/stdx_to_llvm/passes.h"
#include "pmlc/dialect/stdx/ir/ops.h"

namespace pmlc::conversion::stdx_to_llvm {

using namespace mlir; // NOLINT[build/namespaces]

namespace stdx = dialect::stdx;

namespace {

template <typename OpType>
struct LibMCallLowering : public ConvertOpToLLVMPattern<OpType> {
  using ConvertOpToLLVMPattern<OpType>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto f32Type = rewriter.getF32Type();
    SmallVector<Type, 2> argTypes(getArity(), f32Type);
    auto funcType =
        LLVM::LLVMFunctionType::get(f32Type, argTypes, /*isVarArg=*/false);
    auto attr = rewriter.getStringAttr(getFuncName());
    auto sym = getOrInsertFuncOp(attr, funcType, op, rewriter);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, ArrayRef<Type>{f32Type}, SymbolRefAttr::get(attr), adaptor.getOperands());
    return success();
  }

  LLVM::LLVMFuncOp
  getOrInsertFuncOp(StringAttr funcName, LLVM::LLVMFunctionType funcType,
                    Operation *op, ConversionPatternRewriter &rewriter) const {
    Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);

    auto module = op->getParentOfType<ModuleOp>();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    return rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(), funcName.getValue(),
                                             funcType);
  }

protected:
  virtual std::string getFuncName() const = 0;
  virtual size_t getArity() const { return 1; }
};

struct ACosLowering : public LibMCallLowering<stdx::ACosOp> {
  using LibMCallLowering<stdx::ACosOp>::LibMCallLowering;
  std::string getFuncName() const override { return "acosf"; }
};

struct ACosHLowering : public LibMCallLowering<stdx::ACosHOp> {
  using LibMCallLowering<stdx::ACosHOp>::LibMCallLowering;
  std::string getFuncName() const override { return "acoshf"; }
};

struct ASinLowering : public LibMCallLowering<stdx::ASinOp> {
  using LibMCallLowering<stdx::ASinOp>::LibMCallLowering;
  std::string getFuncName() const override { return "asinf"; }
};

struct ASinHLowering : public LibMCallLowering<stdx::ASinHOp> {
  using LibMCallLowering<stdx::ASinHOp>::LibMCallLowering;
  std::string getFuncName() const override { return "asinhf"; }
};

struct ATanLowering : public LibMCallLowering<stdx::ATanOp> {
  using LibMCallLowering<stdx::ATanOp>::LibMCallLowering;
  std::string getFuncName() const override { return "atanf"; }
};

struct ATanHLowering : public LibMCallLowering<stdx::ATanHOp> {
  using LibMCallLowering<stdx::ATanHOp>::LibMCallLowering;
  std::string getFuncName() const override { return "atanhf"; }
};

struct CosHLowering : public LibMCallLowering<stdx::CosHOp> {
  using LibMCallLowering<stdx::CosHOp>::LibMCallLowering;
  std::string getFuncName() const override { return "coshf"; }
};

struct ErfLowering : public LibMCallLowering<stdx::ErfOp> {
  using LibMCallLowering<stdx::ErfOp>::LibMCallLowering;
  std::string getFuncName() const override { return "erff"; }
};

struct FloorLowering : public LibMCallLowering<stdx::FloorOp> {
  using LibMCallLowering<stdx::FloorOp>::LibMCallLowering;
  std::string getFuncName() const override { return "floorf"; }
};

struct PowLowering : public LibMCallLowering<stdx::PowOp> {
  using LibMCallLowering<stdx::PowOp>::LibMCallLowering;
  std::string getFuncName() const override { return "powf"; }
  size_t getArity() const override { return 2; }
};

struct RoundLowering : public LibMCallLowering<stdx::RoundOp> {
  using LibMCallLowering<stdx::RoundOp>::LibMCallLowering;
  std::string getFuncName() const override { return "roundf"; }
};

struct SinHLowering : public LibMCallLowering<stdx::SinHOp> {
  using LibMCallLowering<stdx::SinHOp>::LibMCallLowering;
  std::string getFuncName() const override { return "sinhf"; }
};

struct TanLowering : public LibMCallLowering<stdx::TanOp> {
  using LibMCallLowering<stdx::TanOp>::LibMCallLowering;
  std::string getFuncName() const override { return "tanf"; }
};

class BaseViewConversionHelper {
public:
  BaseViewConversionHelper(OpBuilder &rewriter, Location loc, Type type)
      : rewriter(rewriter), loc(loc),
        desc(MemRefDescriptor::undef(rewriter, loc, type)) {}

  explicit BaseViewConversionHelper(OpBuilder &rewriter, Location loc, Value v)
      : rewriter(rewriter), loc(loc), desc(v) {}

  /// Wrappers around MemRefDescriptor that use EDSC builder and location.
  Value allocatedPtr() { return desc.allocatedPtr(rewriter, loc); }
  void setAllocatedPtr(Value v) { desc.setAllocatedPtr(rewriter, loc, v); }
  Value alignedPtr() { return desc.alignedPtr(rewriter, loc); }
  void setAlignedPtr(Value v) { desc.setAlignedPtr(rewriter, loc, v); }
  Value offset() { return desc.offset(rewriter, loc); }
  void setOffset(Value v) { desc.setOffset(rewriter, loc, v); }
  Value size(unsigned i) { return desc.size(rewriter, loc, i); }
  void setSize(unsigned i, Value v) { desc.setSize(rewriter, loc, i, v); }
  void setConstantSize(unsigned i, int64_t v) {
    desc.setConstantSize(rewriter, loc, i, v);
  }
  Value stride(unsigned i) { return desc.stride(rewriter, loc, i); }
  void setStride(unsigned i, Value v) { desc.setStride(rewriter, loc, i, v); }
  void setConstantStride(unsigned i, int64_t v) {
    desc.setConstantStride(rewriter, loc, i, v);
  }

  operator Value() { return desc; }

private:
  OpBuilder &rewriter;
  Location loc;
  MemRefDescriptor desc;
};

struct ReshapeLowering : public ConvertOpToLLVMPattern<stdx::ReshapeOp> {
  using ConvertOpToLLVMPattern<stdx::ReshapeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(stdx::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MemRefType dstType = op.getResult().getType().cast<MemRefType>();

    if (!dstType.hasStaticShape())
      return failure();

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto res = getStridesAndOffset(dstType, strides, offset);
    if (failed(res) || llvm::any_of(strides, [](int64_t val) {
          return ShapedType::isDynamicStrideOrOffset(val);
        }))
      return failure();

    BaseViewConversionHelper baseDesc(rewriter, op->getLoc(), adaptor.tensor());
    BaseViewConversionHelper desc(rewriter, op->getLoc(),
                                  typeConverter->convertType(dstType));
    desc.setAllocatedPtr(baseDesc.allocatedPtr());
    desc.setAlignedPtr(baseDesc.alignedPtr());
    desc.setOffset(baseDesc.offset());
    for (auto en : llvm::enumerate(dstType.getShape()))
      desc.setConstantSize(en.index(), en.value());
    for (auto en : llvm::enumerate(strides))
      desc.setConstantStride(en.index(), en.value());
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

static LLVM::LLVMFuncOp importFunc(OpBuilder &builder, StringAttr name,
                                   Type funcTy) {
  // Find the enclosing module
  auto moduleOp =
      builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
  // Check if the function is defined
  auto func = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(name);
  if (!func) {
    // If not, make a declaration
    OpBuilder::InsertionGuard insertionGuard{builder};
    builder.setInsertionPointToStart(moduleOp.getBody());
    func = builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(),
                                            name.getValue(), funcTy);
  }
  return func;
}

static LLVM::LLVMStructType getStructType(TypeConverter &converter,
                                          TypeRange types) {
  SmallVector<Type, 8> fieldTypes;
  assert(types.size() > 0);
  for (auto type : types) {
    fieldTypes.push_back(converter.convertType(type));
  }
  return LLVM::LLVMStructType::getLiteral(types[0].getContext(), fieldTypes);
}

struct PackLowering : public ConvertOpToLLVMPattern<stdx::PackOp> {
  using ConvertOpToLLVMPattern<stdx::PackOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(stdx::PackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    if (op.getNumOperands() == 0) {
      auto nullPtr = rewriter.create<LLVM::NullOp>(loc, getVoidPtrType());
      rewriter.replaceOp(op, {nullPtr});
      return success();
    }
    // Get the relevant types
    auto structType = getStructType(*typeConverter, op.getOperandTypes());
    // Get the size of the struct type and malloc
    auto sizeofStruct = getSizeInBytes(loc, structType, rewriter);
    auto mallocFunc =
        importFunc(rewriter, rewriter.getStringAttr("malloc"),
                   LLVM::LLVMFunctionType::get(getVoidPtrType(),
                                               ArrayRef<Type>{getIndexType()},
                                               /*isVarArg=*/false));
    auto rawPtr =
        rewriter.create<LLVM::CallOp>(loc, mallocFunc, ValueRange{sizeofStruct})
            .getResult(0);
    auto structPtr = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(structType), rawPtr);
    // Make a value like struct holding all the fields
    Value structVal = rewriter.create<LLVM::UndefOp>(op.getLoc(), structType);
    for (auto valIdx : llvm::enumerate(adaptor.getOperands())) {
      structVal = rewriter.create<LLVM::InsertValueOp>(
          loc, structType, structVal, valIdx.value(),
          rewriter.getI64ArrayAttr(valIdx.index()));
    }
    // Store the result
    rewriter.create<LLVM::StoreOp>(loc, structVal, structPtr);
    // Return the raw pointer
    rewriter.replaceOp(op, {rawPtr});
    return success();
  }
};

struct UnpackLowering : public ConvertOpToLLVMPattern<stdx::UnpackOp> {
  using ConvertOpToLLVMPattern<stdx::UnpackOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(stdx::UnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.getNumResults() == 0) {
      rewriter.replaceOp(op, {});
      return success();
    }
    Location loc = op.getLoc();
    // Get the LLVM structure type
    auto structType = getStructType(*typeConverter, op.getResultTypes());
    // Bitcast the input operand
    auto structPtr = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(structType), adaptor.getOperands()[0]);
    // Load it
    auto structVal = rewriter.create<LLVM::LoadOp>(loc, structPtr);
    // Extract all the values
    SmallVector<Value, 8> outVals;
    for (unsigned i = 0; i < op.getNumResults(); i++) {
      outVals.push_back(rewriter.create<LLVM::ExtractValueOp>(
          loc, structType.getBody()[i], structVal,
          rewriter.getI64ArrayAttr(i)));
    }
    // Return extracted values
    rewriter.replaceOp(op, outVals);
    return success();
  }
};

/// A pass converting MLIR operations into the LLVM IR dialect.
struct LowerToLLVMPass : public LowerToLLVMBase<LowerToLLVMPass> {
  // Run the dialect converter on the module.
  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    LLVMTypeConverter typeConverter(context);

    RewritePatternSet patterns(context);
    populateLoopToStdConversionPatterns(patterns);
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
    populateMathToLLVMConversionPatterns(typeConverter, patterns);
    populateStdXToLLVMConversionPatterns(typeConverter, patterns);

    LLVMConversionTarget target(*context);
    target.addIllegalDialect<stdx::StdXDialect>();
    target.addIllegalOp<UnrealizedConversionCastOp>();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateStdXToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns) {
  patterns.insert<ACosLowering,    //
                  ACosHLowering,   //
                  ASinLowering,    //
                  ASinHLowering,   //
                  ATanLowering,    //
                  ATanHLowering,   //
                  CosHLowering,    //
                  ErfLowering,     //
                  FloorLowering,   //
                  PackLowering,    //
                  PowLowering,     //
                  ReshapeLowering, //
                  RoundLowering,   //
                  SinHLowering,    //
                  TanLowering,     //
                  UnpackLowering   //
                  >(converter);
  converter.addConversion([](TupleType type) -> Optional<Type> {
    return LLVM::LLVMPointerType::get(IntegerType::get(type.getContext(), 8));
  });
}

std::unique_ptr<Pass> createLowerToLLVMPass() {
  return std::make_unique<LowerToLLVMPass>();
}

} // namespace pmlc::conversion::stdx_to_llvm
