// Copyright 2020, Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/conversion/affine_to_omp/pass_detail.h"
#include "pmlc/conversion/affine_to_omp/passes.h"
#include "pmlc/util/tags.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::conversion::affine_to_omp {

namespace omp = mlir::omp;

namespace {

struct ParallelOpConversion : public OpConversionPattern<AffineParallelOp> {
  using OpConversionPattern<AffineParallelOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AffineParallelOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    bool isThread = hasUnitTag(op, cpuBlockTag());
    if (isThread) {
      // snip: broken
      return failure();
    } else {
      return success();
    }
  }
};

/// A pass converting affine operations into the OpenMP dialect.
struct LowerAffineToOpenMPPass
    : public LowerAffineToOpenMPBase<LowerAffineToOpenMPPass> {
  // Run the dialect converter on the module.
  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto context = module.getContext();

    OwningRewritePatternList patterns;
    populateAffineToOpenMPConversionPatterns(patterns, context);

    ConversionTarget target(*context);
    target.addLegalDialect<omp::OpenMPDialect>();
    if (failed(applyPartialConversion(module, target, patterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateAffineToOpenMPConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ParallelOpConversion>(ctx);
}

std::unique_ptr<mlir::Pass> createLowerAffineToOpenMPPass() {
  return std::make_unique<LowerAffineToOpenMPPass>();
}

} // namespace pmlc::conversion::affine_to_omp
