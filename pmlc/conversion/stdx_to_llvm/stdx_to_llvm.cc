// Copyright 2020, Intel Corporation

#include "pmlc/conversion/stdx_to_llvm/stdx_to_llvm.h"

#include "mlir/ADT/TypeSwitch.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/stdx/ir/ops.h"

using namespace mlir;  // NOLINT[build/namespaces]

namespace pmlc::conversion::stdx_to_llvm {

namespace stdx = dialect::stdx;

namespace {

template <typename SourceOp>
class LLVMLegalizationPattern : public LLVMOpLowering {
 public:
  explicit LLVMLegalizationPattern(LLVMTypeConverter& converter)
      : LLVMOpLowering(SourceOp::getOperationName(), converter.getDialect()->getContext(), converter) {}

 protected:
  LLVM::LLVMDialect& getDialect() const { return lowering.getDialect(); }
};

Optional<LLVM::AtomicBinOp> matchBinOp(stdx::AtomicRMWOp op) {
  auto* body = op.getBody();
  auto* terminator = body->getTerminator();
  auto yieldOp = llvm::cast<stdx::AtomicRMWYieldOp>(terminator);
  auto defOp = yieldOp.result().getDefiningOp();
  return TypeSwitch<Operation*, Optional<LLVM::AtomicBinOp>>(defOp)
      .Case<AddFOp>([](AddFOp op) { return LLVM::AtomicBinOp::fadd; })
      .Case<SubFOp>([](SubFOp op) { return LLVM::AtomicBinOp::fsub; })
      .Case<AddIOp>([](AddIOp op) { return LLVM::AtomicBinOp::add; })
      .Case<SubIOp>([](SubIOp op) { return LLVM::AtomicBinOp::sub; })
      .Default([](Operation* op) { return llvm::None; });
}

struct AtomicRMWOpLowering : public LLVMLegalizationPattern<stdx::AtomicRMWOp> {
  using LLVMLegalizationPattern<stdx::AtomicRMWOp>::LLVMLegalizationPattern;

  PatternMatchResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                     ConversionPatternRewriter& rewriter) const override {
    auto atomicOp = llvm::cast<stdx::AtomicRMWOp>(op);
    auto binOp = matchBinOp(atomicOp);
    if (!binOp.hasValue()) {
      return matchFailure();
    }
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

struct CmpXchgLowering : public LLVMLegalizationPattern<stdx::AtomicRMWOp> {
  using LLVMLegalizationPattern<stdx::AtomicRMWOp>::LLVMLegalizationPattern;

  PatternMatchResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                     ConversionPatternRewriter& rewriter) const override {
    auto atomicOp = llvm::cast<stdx::AtomicRMWOp>(op);
    auto binOp = matchBinOp(atomicOp);
    if (binOp.hasValue()) {
      return matchFailure();
    }
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

void populateStdXToLLVMConversionPatterns(LLVMTypeConverter& converter, OwningRewritePatternList& patterns) {
  patterns.insert<AtomicRMWOpLowering, CmpXchgLowering>(converter);
}

/// A pass converting MLIR operations into the LLVM IR dialect.
struct LLVMLoweringPass : public ModulePass<LLVMLoweringPass> {
  // Run the dialect converter on the module.
  void runOnModule() override {
    ModuleOp module = getModule();
    auto context = module.getContext();
    LLVM::ensureDistinctSuccessors(module);
    LLVMTypeConverter typeConverter(context);

    OwningRewritePatternList patterns;
    populateLoopToStdConversionPatterns(patterns, context);
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    populateStdXToLLVMConversionPatterns(typeConverter, patterns);

    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    if (failed(applyPartialConversion(module, target, patterns, &typeConverter))) {
      signalPassFailure();
    }
  }
};

static PassRegistration<LLVMLoweringPass> pass("convert-stdx-to-llvm", "Convert stdx to the LLVM dialect");

}  // namespace

std::unique_ptr<mlir::Pass> createLowerToLLVMPass() { return std::make_unique<LLVMLoweringPass>(); }

}  // namespace pmlc::conversion::stdx_to_llvm
