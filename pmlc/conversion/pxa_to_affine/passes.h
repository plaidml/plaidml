// Copyright 2020, Intel Corporation

#pragma once

#include <memory>

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::conversion::pxa_to_affine {

class PXAToAffineConversionTarget : public mlir::ConversionTarget {
public:
  explicit PXAToAffineConversionTarget(mlir::MLIRContext &ctx);
};

void populatePXAToAffineConversionPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *ctx);

std::unique_ptr<mlir::Pass> createLowerPXAToAffinePass();

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/conversion/pxa_to_affine/passes.h.inc"

} // namespace pmlc::conversion::pxa_to_affine
