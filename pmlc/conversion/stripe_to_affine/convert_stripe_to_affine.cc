// Copyright 2019 Intel Corporation

#include "pmlc/conversion/stripe_to_affine/convert_stripe_to_affine.h"
#include "pmlc/dialect/stripe/ops.h"

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define PASS_NAME "convert-stripe-to-affine"
#define DEBUG_TYPE PASS_NAME

namespace {

// Pass to convert Stripe dialect to Affine dialect.
struct ConvertStripeToAffine : public mlir::FunctionPass<ConvertStripeToAffine> {
  void runOnFunction() override;
};

void ConvertStripeToAffine::runOnFunction() {
  mlir::OwningRewritePatternList patterns;
  pmlc::conversion::stripe_to_affine::populateStripeToAffineConversionPatterns(patterns, &getContext());

  // Add Affine/Std dialect legal ops to conversion target.
  mlir::ConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
  target.addLegalDialect<mlir::AffineOpsDialect, mlir::StandardOpsDialect>();
  target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
    // TODO: FuncOp is legal only if types have been converted to Std types.
    return true;  // typeConverter.isSignatureLegal(op.getType());
  });

  if (failed(mlir::applyFullConversion(getFunction(), target, patterns))) {
    signalPassFailure();
  }
}

}  // namespace

namespace pmlc {
namespace conversion {
namespace stripe_to_affine {

using mlir::ArrayAttr;
using mlir::isa;
using mlir::OpRewritePattern;
using mlir::PatternMatchResult;
using mlir::PatternRewriter;
using mlir::SmallVector;
using pmlc::dialect::stripe::AffinePolyOp;
using pmlc::dialect::stripe::ParallelForOp;
using pmlc::dialect::stripe::TerminateOp;

// Declaration of Stripe ops converters for supported ops.
#define STRIPE_OP(OP)                                                                    \
  struct OP##Converter : public OpRewritePattern<OP> {                                   \
    using OpRewritePattern<OP>::OpRewritePattern;                                        \
                                                                                         \
    PatternMatchResult matchAndRewrite(OP Op, PatternRewriter& rewriter) const override; \
  };
#include "supported_ops.inc"

PatternMatchResult AffinePolyOpConverter::matchAndRewrite(AffinePolyOp constOp, PatternRewriter& rewriter) const {
  // TODO: This is no longer correct
  rewriter.replaceOpWithNewOp<mlir::ConstantIndexOp>(constOp, constOp.offset().getSExtValue());
  return matchSuccess();
}

PatternMatchResult ParallelForOpConverter::matchAndRewrite(ParallelForOp forOp, PatternRewriter& rewriter) const {
  auto forRanges = forOp.ranges().getValue();
  auto& forBodyRegion = forOp.inner();
  assert(forBodyRegion.getBlocks().size() == 1 && "Unexpected control flow in Stripe");

  if (!forRanges.size()) {
    // This is ParallelForOp with no ranges so no affine.loop needed ("main" case). Move ParallelForOp's operations
    // (single block) into parent single block. ParallelForOp terminator is not moved since parent region already have
    // one.
    auto& parentBlockOps = forOp.getOperation()->getBlock()->getOperations();
    auto& forBodyOps = forBodyRegion.front().getOperations();
    assert(isa<TerminateOp>(parentBlockOps.back()) && "Expected terminator");
    parentBlockOps.splice(mlir::Block::iterator(forOp), forBodyOps, forBodyOps.begin(), std::prev(forBodyOps.end()));
  } else {
    llvm_unreachable("ParallelForOp not supported yet");
  }

  // We are done. Remove ParallelForOp.
  rewriter.replaceOp(forOp, {});
  return matchSuccess();
}

// Converts TerminateOp to AffineTerminatorOp. If a TerminateOp requires a special context-dependent treatment, that
// must be implemented in the Op providing the context.
PatternMatchResult TerminateOpConverter::matchAndRewrite(TerminateOp terminateOp, PatternRewriter& rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::AffineTerminatorOp>(terminateOp);
  return matchSuccess();
}

void populateStripeToAffineConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* ctx) {
#define STRIPE_OP(OP) OP##Converter,
#define STRIPE_LAST_OP(OP) OP##Converter
  patterns.insert<
#include "supported_ops.inc"  // NOLINT(build/include)
      >(ctx);
}

}  // namespace stripe_to_affine
}  // namespace conversion
}  // namespace pmlc

std::unique_ptr<mlir::FunctionPassBase> mlir::createConvertStripeToAffinePass() {
  return std::make_unique<ConvertStripeToAffine>();
}

static mlir::PassRegistration<ConvertStripeToAffine> pass(PASS_NAME, "Convert Stripe dialect to Affine dialect");
