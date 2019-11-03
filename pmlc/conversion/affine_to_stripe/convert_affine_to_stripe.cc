// Copyright 2019 Intel Corporation

#include "pmlc/conversion/affine_to_stripe/convert_affine_to_stripe.h"
#include "pmlc/dialect/stripe/affine_poly.h"
#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/ops.h"
#include "pmlc/dialect/stripe/util.h"

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define PASS_NAME "convert-affine-to-stripe"
#define DEBUG_TYPE PASS_NAME

namespace {

// Pass to convert Affine dialect to Stripe dialect.
struct ConvertAffineToStripe : public mlir::FunctionPass<ConvertAffineToStripe> {
  void runOnFunction() override;
};

void ConvertAffineToStripe::runOnFunction() {
  mlir::OwningRewritePatternList patterns;
  pmlc::conversion::affine_to_stripe::populateAffineToStripeConversionPatterns(patterns, &getContext());

  // Add Stripe dialect legal ops to conversion target.
  mlir::ConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
  target.addLegalDialect<pmlc::dialect::stripe::Dialect>();
  target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
    // TODO: FuncOp is legal only if types have been converted to Eltwise types.
    return true;  // typeConverter.isSignatureLegal(op.getType());
  });

  auto funcOp = getFunction();
  if (failed(mlir::applyFullConversion(funcOp, target, patterns))) {
    signalPassFailure();
  }

  // Wrap function body with a ParallelForOp to represent Stripe's 'main' block.
  pmlc::dialect::stripe::createMainParallelFor(funcOp);
}

}  // namespace

namespace pmlc {
namespace conversion {
namespace affine_to_stripe {

using mlir::AffineForOp;
using mlir::AffineTerminatorOp;
using mlir::ConstantIndexOp;
using mlir::OpRewritePattern;
using mlir::PatternMatchResult;
using mlir::PatternRewriter;
using pmlc::dialect::stripe::AffinePolynomial;
using pmlc::dialect::stripe::AffinePolyOp;
using pmlc::dialect::stripe::TerminateOp;

// Declaration of Affine ops converters for supported ops.
#define AFFINE_OP(OP)                                                                    \
  struct OP##Converter : public OpRewritePattern<OP> {                                   \
    using OpRewritePattern<OP>::OpRewritePattern;                                        \
                                                                                         \
    PatternMatchResult matchAndRewrite(OP Op, PatternRewriter& rewriter) const override; \
  };
#include "supported_ops.inc"

PatternMatchResult ConstantIndexOpConverter::matchAndRewrite(ConstantIndexOp constOp, PatternRewriter& rewriter) const {
  int64_t val = constOp.value().cast<mlir::IntegerAttr>().getInt();
  // AffinePolyOp currently has a fixed 64-bit integer type.
  rewriter.replaceOpWithNewOp<AffinePolyOp>(constOp, AffinePolynomial(val));
  return matchSuccess();
}

PatternMatchResult AffineForOpConverter::matchAndRewrite(AffineForOp forOp, PatternRewriter& rewriter) const {
  llvm_unreachable("ParallelForOp not supported yet");
}

// Converts AffineTerminatorOp to TerminateOp. If an AffineTerminatorOp requires a special context-dependent treatment,
// that must be implemented in the Op providing the context.
PatternMatchResult AffineTerminatorOpConverter::matchAndRewrite(AffineTerminatorOp terminatorOp,
                                                                PatternRewriter& rewriter) const {
  rewriter.replaceOpWithNewOp<TerminateOp>(terminatorOp);
  return matchSuccess();
}

void populateAffineToStripeConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* ctx) {
#define AFFINE_OP(OP) OP##Converter,
#define AFFINE_LAST_OP(OP) OP##Converter
  patterns.insert<
#include "supported_ops.inc"  // NOLINT(build/include)
      >(ctx);
}

}  // namespace affine_to_stripe
}  // namespace conversion
}  // namespace pmlc

std::unique_ptr<mlir::FunctionPassBase> mlir::createConvertAffineToStripePass() {
  return std::make_unique<ConvertAffineToStripe>();
}

static mlir::PassRegistration<ConvertAffineToStripe> pass(PASS_NAME, "Convert Affine dialect to Stripe dialect");
