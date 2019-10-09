// Copyright 2019 Intel Corporation

#include "pmlc/conversion/stripe_to_affine/convert_stripe_to_affine.h"

#include <mlir/Dialect/AffineOps/AffineOps.h>
#include <mlir/Dialect/StandardOps/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#define PASS_NAME "convert-stripe-to-affine"
#define DEBUG_TYPE PASS_NAME

namespace {

// Pass to convert Stripe dialect to Affine dialect.
struct ConvertStripeToAffine : public mlir::FunctionPass<ConvertStripeToAffine> {
  void runOnFunction() override;
};

void ConvertStripeToAffine::runOnFunction() {
  mlir::OwningRewritePatternList patterns;
  pmlc::populateStripeToAffineConversionPatterns(patterns, &getContext());
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::AffineOpsDialect, mlir::StandardOpsDialect>();

  if (failed(mlir::applyPartialConversion(getFunction(), target, patterns))) signalPassFailure();
}
}  // namespace

void pmlc::populateStripeToAffineConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* ctx) {
  // patterns.insert<ForLowering, IfLowering, TerminatorLowering>(ctx);
}

std::unique_ptr<mlir::FunctionPassBase> mlir::createConvertStripeToAffinePass() {
  return std::make_unique<ConvertStripeToAffine>();
}

static mlir::PassRegistration<ConvertStripeToAffine> pass(PASS_NAME, "Convert Stripe dialect to affine dialect");
