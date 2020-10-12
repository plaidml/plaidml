// Copyright 2020, Intel Corporation

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/conversion/scf_to_omp/pass_detail.h"
#include "pmlc/conversion/scf_to_omp/passes.h"
#include "pmlc/util/tags.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::conversion::scf_to_omp {

namespace scf = mlir::scf;
namespace omp = mlir::omp;

namespace {

struct ParallelOpConversion : public OpConversionPattern<scf::ParallelOp> {
  using OpConversionPattern<scf::ParallelOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ParallelOp op, ArrayRef<Value> operands,
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

/// A pass converting SCF parallel loop into the OpenMP dialect.
struct LowerSCFToOpenMPPass
    : public LowerSCFToOpenMPBase<LowerSCFToOpenMPPass> {
  // Run the dialect converter on the module.
  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto context = module.getContext();

    OwningRewritePatternList patterns;
    populateSCFToOpenMPConversionPatterns(patterns, context);

    ConversionTarget target(*context);
    target.addLegalDialect<omp::OpenMPDialect>();
    if (failed(applyPartialConversion(module, target, patterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateSCFToOpenMPConversionPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *ctx) {
  patterns.insert<ParallelOpConversion>(ctx);
}

std::unique_ptr<mlir::Pass> createLowerSCFToOpenMPPass() {
  return std::make_unique<LowerSCFToOpenMPPass>();
}

} // namespace pmlc::conversion::scf_to_omp
