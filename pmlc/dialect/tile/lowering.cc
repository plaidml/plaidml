// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/lowering.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "base/util/logging.h"
#include "pmlc/dialect/eltwise/dialect.h"
#include "pmlc/dialect/eltwise/util.h"
#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/ops.h"
#include "pmlc/dialect/tile/internal.h"
#include "pmlc/dialect/tile/ops.h"

namespace pmlc {
namespace dialect {
namespace tile {

class AffineConstantOpConversion : public mlir::ConversionPattern {
 public:
  explicit AffineConstantOpConversion(mlir::MLIRContext* context)  //
      : ConversionPattern(AffineConstantOp::getOperationName(), 1, context) {}

  mlir::PatternMatchResult matchAndRewrite(   //
      mlir::Operation* op,                    //
      llvm::ArrayRef<mlir::Value*> operands,  //
      mlir::ConversionPatternRewriter& rewriter) const override {
    IVLOG(1, "AffineConstantOpConversion::matchAndRewrite>");
    auto constOp = llvm::cast<AffineConstantOp>(op);
    auto newOp = rewriter.create<stripe::AffineConstOp>(  //
        op->getLoc(),                                     //
        rewriter.getType<stripe::AffineType>(),           //
        rewriter.getI64IntegerAttr(constOp.value().getSExtValue()));
    rewriter.replaceOp(op, {newOp});
    return matchSuccess();
  }
};

class AffineDomainOpConversion : public mlir::ConversionPattern {
 public:
  explicit AffineDomainOpConversion(mlir::MLIRContext* context)  //
      : ConversionPattern(AffineDomainOp::getOperationName(), 1, context) {}

  mlir::PatternMatchResult matchAndRewrite(   //
      mlir::Operation* op,                    //
      llvm::ArrayRef<mlir::Value*> operands,  //
      mlir::ConversionPatternRewriter& rewriter) const override {
    IVLOG(1, "AffineDomainOpConversion::matchAndRewrite>");
    // TODO
    // 1. Construct a contraction from a DomainOp
    // 2. Compile contraction:
    //    1. ConstrainIndexVarsToInts
    //    2. GatherConstraints
    //    3. ReduceOutputPolynomials
    //    4. MergeParallelConstraints
    //    5. Defract
    //    6. GatherConstraints
    //    7. MergeParallelConstraints
    //    8. ComputeBounds
    return matchSuccess();
  }
};

struct LoweringPass : public mlir::ModulePass<LoweringPass> {
  void runOnModule() override {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<eltwise::Dialect, stripe::Dialect>();
    mlir::OwningRewritePatternList patterns;
    patterns.insert<AffineConstantOpConversion, AffineDomainOpConversion>(&getContext());
    if (failed(applyPartialConversion(getModule(), target, patterns))) {
      emitError(mlir::UnknownLoc::get(&getContext()), "Error lowering tile -> stripe\n");
      signalPassFailure();
    }
  }

  static std::unique_ptr<mlir::Pass> Create() { return std::make_unique<LoweringPass>(); }
};

mlir::OwningModuleRef LowerIntoStripe(mlir::MLIRContext* context, TileProgram* program) {
  IVLOG(1, "LowerIntoStripe");
  mlir::OwningModuleRef module(llvm::cast<mlir::ModuleOp>(program->module->getOperation()->clone()));
  mlir::PassManager pm;
  IVLOG(1, "before:" << *module);
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(LoweringPass::Create());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  auto result = pm.run(*module);
  if (failed(result)) {
    IVLOG(1, "LowerIntoStripe failed: " << *module);
    throw std::runtime_error("Lowering to stripe dialect failure");
  }
  IVLOG(1, "after:" << *module);
  return module;
}

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
