// Copyright 2020, Intel Corporation

#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/conversion/stdx_to_llvm/pass_detail.h"
#include "pmlc/conversion/stdx_to_llvm/passes.h"
#include "pmlc/dialect/stdx/ir/ops.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::conversion::stdx_to_llvm {

namespace stdx = dialect::stdx;
namespace edsc = mlir::edsc;
using LLVMType = LLVM::LLVMType;
using LLVMStructType = LLVM::LLVMStructType;

namespace {

template <typename T>
struct LibMCallLowering : public ConvertOpToLLVMPattern<T> {
  using ConvertOpToLLVMPattern<T>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto f32Type = LLVM::LLVMType::getFloatTy(rewriter.getContext());
    SmallVector<LLVM::LLVMType, 2> argTypes(getArity(), f32Type);
    auto funcType = LLVM::LLVMType::getFunctionTy(f32Type, argTypes, false);
    auto sym = getOrInsertFuncOp(getFuncName(), funcType, op);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, ArrayRef<Type>{f32Type}, rewriter.getSymbolRefAttr(sym), operands);
    return success();
  }

  LLVM::LLVMFuncOp getOrInsertFuncOp(StringRef funcName,
                                     LLVM::LLVMType funcType,
                                     Operation *op) const {
    using LLVM::LLVMFuncOp;

    Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcName);
    if (funcOp)
      return cast<LLVMFuncOp>(*funcOp);

    mlir::OpBuilder builder(op->getParentOfType<LLVMFuncOp>());
    return builder.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
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
  explicit BaseViewConversionHelper(Type type)
      : desc(MemRefDescriptor::undef(rewriter(), loc(), type)) {}

  explicit BaseViewConversionHelper(Value v) : desc(v) {}

  /// Wrappers around MemRefDescriptor that use EDSC builder and location.
  Value allocatedPtr() { return desc.allocatedPtr(rewriter(), loc()); }
  void setAllocatedPtr(Value v) { desc.setAllocatedPtr(rewriter(), loc(), v); }
  Value alignedPtr() { return desc.alignedPtr(rewriter(), loc()); }
  void setAlignedPtr(Value v) { desc.setAlignedPtr(rewriter(), loc(), v); }
  Value offset() { return desc.offset(rewriter(), loc()); }
  void setOffset(Value v) { desc.setOffset(rewriter(), loc(), v); }
  Value size(unsigned i) { return desc.size(rewriter(), loc(), i); }
  void setSize(unsigned i, Value v) { desc.setSize(rewriter(), loc(), i, v); }
  void setConstantSize(unsigned i, int64_t v) {
    desc.setConstantSize(rewriter(), loc(), i, v);
  }
  Value stride(unsigned i) { return desc.stride(rewriter(), loc(), i); }
  void setStride(unsigned i, Value v) {
    desc.setStride(rewriter(), loc(), i, v);
  }
  void setConstantStride(unsigned i, int64_t v) {
    desc.setConstantStride(rewriter(), loc(), i, v);
  }

  operator Value() { return desc; }

private:
  OpBuilder &rewriter() { return edsc::ScopedContext::getBuilderRef(); }
  Location loc() { return edsc::ScopedContext::getLocation(); }

  MemRefDescriptor desc;
};

struct ReshapeLowering : public ConvertOpToLLVMPattern<stdx::ReshapeOp> {
  using ConvertOpToLLVMPattern<stdx::ReshapeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto reshapeOp = cast<stdx::ReshapeOp>(op);
    MemRefType dstType = reshapeOp.getResult().getType().cast<MemRefType>();

    if (!dstType.hasStaticShape())
      return failure();

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto res = getStridesAndOffset(dstType, strides, offset);
    if (failed(res) || llvm::any_of(strides, [](int64_t val) {
          return ShapedType::isDynamicStrideOrOffset(val);
        }))
      return failure();

    edsc::ScopedContext context(rewriter, op->getLoc());
    stdx::ReshapeOpAdaptor adaptor(operands);
    BaseViewConversionHelper baseDesc(adaptor.tensor());
    BaseViewConversionHelper desc(typeConverter.convertType(dstType));
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

static LLVM::LLVMFuncOp importFunc(OpBuilder &builder, mlir::StringRef name,
                                   LLVMType funcTy) {
  // Find the enclosing module
  auto moduleOp =
      builder.getBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>();
  // Check if the function is defined
  auto func = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(name);
  if (!func) {
    // If not, make a declaration
    mlir::OpBuilder::InsertionGuard insertionGuard{builder};
    builder.setInsertionPointToStart(moduleOp.getBody());
    func =
        builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), name, funcTy);
  }
  return func;
}

static LLVMStructType getStructType(TypeConverter &converter, TypeRange types) {
  mlir::SmallVector<LLVMType, 8> fieldTypes;
  assert(types.size() > 0);
  for (auto type : types) {
    fieldTypes.push_back(converter.convertType(type).cast<LLVMType>());
  }
  return LLVMType::getStructTy(types[0].getContext(), fieldTypes)
      .cast<LLVMStructType>();
}

struct PackLowering : public ConvertOpToLLVMPattern<stdx::PackOp> {
  using ConvertOpToLLVMPattern<stdx::PackOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Operation *baseOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto op = cast<stdx::PackOp>(baseOp);
    Location loc = op.getLoc();
    auto i8PtrType = LLVMType::getInt8Ty(rewriter.getContext()).getPointerTo();
    if (op.getNumOperands() == 0) {
      auto nullPtr = rewriter.create<LLVM::NullOp>(loc, i8PtrType);
      rewriter.replaceOp(op, {nullPtr});
      return success();
    }
    // Get the relevant types
    auto structType = getStructType(typeConverter, op.getOperandTypes());
    // Get the size of the struct type and malloc
    auto sizeofStruct = getSizeInBytes(loc, structType, rewriter);
    auto mallocFunc =
        importFunc(rewriter, "malloc",
                   LLVMType::getFunctionTy(
                       i8PtrType, mlir::ArrayRef<LLVMType>{getIndexType()},
                       /*isVarArg=*/false));
    auto rawPtr = rewriter
                      .create<LLVM::CallOp>(loc, mallocFunc,
                                            mlir::ValueRange{sizeofStruct})
                      .getResult(0);
    auto structPtr = rewriter.create<LLVM::BitcastOp>(
        loc, structType.getPointerTo(), rawPtr);
    // Make a value like struct holding all the fields
    mlir::Value structVal =
        rewriter.create<LLVM::UndefOp>(op.getLoc(), structType);
    for (auto valIdx : llvm::enumerate(operands)) {
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
  matchAndRewrite(Operation *baseOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto op = cast<stdx::UnpackOp>(baseOp);
    if (op.getNumResults() == 0) {
      rewriter.replaceOp(op, {});
      return success();
    }
    Location loc = op.getLoc();
    // Get the LLVM structure type
    auto structType = getStructType(typeConverter, op.getResultTypes());
    // Bitcast the input operand
    auto structPtr = rewriter.create<LLVM::BitcastOp>(
        loc, structType.getPointerTo(), operands[0]);
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
    auto context = module.getContext();
    LLVMTypeConverter typeConverter(context);

    OwningRewritePatternList patterns;
    populateLoopToStdConversionPatterns(patterns, context);
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    populateStdXToLLVMConversionPatterns(typeConverter, patterns);

    LLVMConversionTarget target(*context);
    target.addIllegalDialect<stdx::StdXDialect>();
    if (failed(applyPartialConversion(module, target, patterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateStdXToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          OwningRewritePatternList &patterns) {
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
  converter.addConversion([](stdx::ArgpackType type) -> Optional<Type> {
    // Argpack types look like i8 pointers.  I'd like this to be void*, but MLIR
    // disallows void* in it's validation for some bizare reason
    return LLVMType::getInt8Ty(type.getContext()).getPointerTo();
  });
}

std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
  return std::make_unique<LowerToLLVMPass>();
}

} // namespace pmlc::conversion::stdx_to_llvm
