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

/// Creates a pass to convert Stripe dialect to affine dialect.
std::unique_ptr<FunctionPassBase> createConvertStripeToAffinePass();
}  // namespace mlir

namespace pmlc {

/// Collect a set of patterns to convert Stripe dialect ops into affine dialect ops.
void populateStripeToAffineConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* ctx);

}  // namespace pmlc
