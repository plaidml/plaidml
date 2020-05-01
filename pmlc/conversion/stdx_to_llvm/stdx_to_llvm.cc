// Copyright 2020, Intel Corporation

#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
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

namespace {

// Base class for Standard to LLVM IR op conversions.  Matches the Op type
// provided as template argument.  Carries a reference to the LLVM dialect in
// case it is necessary for rewriters.
template <typename SourceOp>
class LLVMLegalizationPattern : public ConvertToLLVMPattern {
public:
  // Construct a conversion pattern.
  explicit LLVMLegalizationPattern(LLVM::LLVMDialect &dialect_,
                                   LLVMTypeConverter &typeConverter_)
      : ConvertToLLVMPattern(SourceOp::getOperationName(),
                             dialect_.getContext(), typeConverter_),
        dialect(dialect_) {}

  // Get the LLVM IR dialect.
  LLVM::LLVMDialect &getDialect() const { return dialect; }

  // Get the LLVM context.
  llvm::LLVMContext &getContext() const { return dialect.getLLVMContext(); }

  // Get the LLVM module in which the types are constructed.
  llvm::Module &getOperation() const { return dialect.getLLVMModule(); }

  // Get the MLIR type wrapping the LLVM integer type whose bit width is defined
  // by the pointer size used in the LLVM module.
  LLVM::LLVMType getIndexType() const {
    return LLVM::LLVMType::getIntNTy(
        &dialect, getOperation().getDataLayout().getPointerSizeInBits());
  }

  LLVM::LLVMType getVoidType() const {
    return LLVM::LLVMType::getVoidTy(&dialect);
  }

  // Get the MLIR type wrapping the LLVM i8* type.
  LLVM::LLVMType getVoidPtrType() const {
    return LLVM::LLVMType::getInt8PtrTy(&dialect);
  }

  // Create an LLVM IR pseudo-operation defining the given index constant.
  Value createIndexConstant(ConversionPatternRewriter &builder, Location loc,
                            uint64_t value) const {
    return createIndexAttrConstant(builder, loc, getIndexType(), value);
  }

protected:
  LLVM::LLVMDialect &dialect;
};

struct FPToSILowering : public LLVMLegalizationPattern<stdx::FPToSIOp> {
  using LLVMLegalizationPattern<stdx::FPToSIOp>::LLVMLegalizationPattern;
  using Base = LLVMLegalizationPattern<stdx::FPToSIOp>;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = op->getOperand(0);
    auto stdxType = op->getResult(0).getType();
    auto llvmType = typeConverter.convertType(stdxType);
    rewriter.replaceOpWithNewOp<LLVM::FPToSIOp>(op, llvmType, value);
    return success();
  }
};

struct FPToUILowering : public LLVMLegalizationPattern<stdx::FPToUIOp> {
  using LLVMLegalizationPattern<stdx::FPToUIOp>::LLVMLegalizationPattern;
  using Base = LLVMLegalizationPattern<stdx::FPToUIOp>;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = op->getOperand(0);
    auto stdxType = op->getResult(0).getType();
    auto llvmType = typeConverter.convertType(stdxType);
    rewriter.replaceOpWithNewOp<LLVM::FPToUIOp>(op, llvmType, value);
    return success();
  }
};

class BaseViewConversionHelper {
public:
  explicit BaseViewConversionHelper(Type type)
      : d(MemRefDescriptor::undef(rewriter(), loc(), type)) {}

  explicit BaseViewConversionHelper(Value v) : d(v) {}

  /// Wrappers around MemRefDescriptor that use EDSC builder and location.
  Value allocatedPtr() { return d.allocatedPtr(rewriter(), loc()); }
  void setAllocatedPtr(Value v) { d.setAllocatedPtr(rewriter(), loc(), v); }
  Value alignedPtr() { return d.alignedPtr(rewriter(), loc()); }
  void setAlignedPtr(Value v) { d.setAlignedPtr(rewriter(), loc(), v); }
  Value offset() { return d.offset(rewriter(), loc()); }
  void setOffset(Value v) { d.setOffset(rewriter(), loc(), v); }
  Value size(unsigned i) { return d.size(rewriter(), loc(), i); }
  void setSize(unsigned i, Value v) { d.setSize(rewriter(), loc(), i, v); }
  void setConstantSize(unsigned i, int64_t v) {
    d.setConstantSize(rewriter(), loc(), i, v);
  }
  Value stride(unsigned i) { return d.stride(rewriter(), loc(), i); }
  void setStride(unsigned i, Value v) { d.setStride(rewriter(), loc(), i, v); }
  void setConstantStride(unsigned i, int64_t v) {
    d.setConstantStride(rewriter(), loc(), i, v);
  }

  operator Value() { return d; }

private:
  OpBuilder &rewriter() { return edsc::ScopedContext::getBuilder(); }
  Location loc() { return edsc::ScopedContext::getLocation(); }

  MemRefDescriptor d;
};

struct ReshapeLowering : public LLVMLegalizationPattern<stdx::ReshapeOp> {
  using LLVMLegalizationPattern<stdx::ReshapeOp>::LLVMLegalizationPattern;
  using Base = LLVMLegalizationPattern<stdx::ReshapeOp>;

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
    stdx::ReshapeOpOperandAdaptor adaptor(operands);
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

    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    if (failed(
            applyPartialConversion(module, target, patterns, &typeConverter))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateStdXToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          OwningRewritePatternList &patterns) {
  patterns.insert<FPToSILowering, FPToUILowering, ReshapeLowering>(
      *converter.getDialect(), converter);
}

std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
  return std::make_unique<LowerToLLVMPass>();
}

} // namespace pmlc::conversion::stdx_to_llvm
