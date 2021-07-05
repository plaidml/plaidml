// Copyright 2020, Intel Corporation

#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/pass_detail.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::tile {

namespace {

// From:
// %zero = tile.constant(0.0 : f64)
// %0 = tile.contract add, mul, %zero, %I, %K
// %1 = tile.add %0, %B
// %2 = tile.relu %1
// Into:
// %0 = tile.contract add, mul, %B, %I, %K
// %2 = tile.relu %0
struct AddInitPattern final : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const {
    if (ContractionOp contractOp =
            dyn_cast_or_null<ContractionOp>(op.lhs().getDefiningOp())) {
      if (!contractOp->hasOneUse())
        return failure();
      if (isa_and_nonnull<ContractionOp>(op.rhs().getDefiningOp()))
        return failure();
      if (op.result().getType() != contractOp.result().getType())
        return failure();

      FloatAttr init;
      if (!matchPattern(contractOp.init(), m_Constant(&init)))
        return failure();
      if (init.getValueAsDouble() != 0.0)
        return failure();

      contractOp.setOperand(0, op.rhs());
      rewriter.replaceOp(op, contractOp.result());

      return success();
    }
    return failure();
  }
};

struct AlgebraicOptPass : public AlgebraicOptBase<AlgebraicOptPass> {
  void runOnFunction() final {
    FuncOp op = getFunction();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.insert<AddInitPattern>(context);

    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      op.emitOpError("AlgebraicOpt pass failure.");
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> createAlgebraicOptPass() {
  return std::make_unique<AlgebraicOptPass>();
}

} // namespace pmlc::dialect::tile
