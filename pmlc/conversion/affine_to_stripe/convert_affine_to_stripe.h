// Copyright 2019 Intel Corporation

#pragma once

#include <memory>

namespace mlir {

class FuncOp;
class MLIRContext;
template <typename T>
class OpPassBase;
using FunctionPassBase = OpPassBase<FuncOp>;
class OwningRewritePatternList;

/// Creates a pass to convert Affine dialect to Stripe dialect.
std::unique_ptr<FunctionPassBase> createConvertAffineToStripePass();

}  // namespace mlir

namespace pmlc {
namespace conversion {
namespace affine_to_stripe {

/// Collect a set of patterns to convert Affine dialect ops to Stripe dialect ops.
void populateAffineToStripeConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* ctx);

}  // namespace affine_to_stripe
}  // namespace conversion
}  // namespace pmlc
