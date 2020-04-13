// Copyright 2020, Intel Corporation

#include "pmlc/conversion/stdx_to_llvm/stdx_to_llvm.h"

#include "mlir/ADT/TypeSwitch.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/stdx/ir/ops.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::conversion::stdx_to_llvm {

namespace stdx = dialect::stdx;

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

/// A pass converting MLIR operations into the LLVM IR dialect.
struct LLVMLoweringPass
    : public mlir::PassWrapper<LLVMLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  // Run the dialect converter on the module.
  void runOnOperation() override {
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

static PassRegistration<LLVMLoweringPass>
    pass("convert-stdx-to-llvm", "Convert stdx to the LLVM dialect");

} // namespace

void populateStdXToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          OwningRewritePatternList &patterns) {
  patterns.insert<FPToSILowering, FPToUILowering>(*converter.getDialect(),
                                                  converter);
}

std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
  return std::make_unique<LLVMLoweringPass>();
}

} // namespace pmlc::conversion::stdx_to_llvm
