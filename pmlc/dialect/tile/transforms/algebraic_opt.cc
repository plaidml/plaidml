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
    return success(succeeded(matchAndRewritePermutation(op, rewriter, op.lhs(),
                                                        op.rhs())) ||
                   succeeded(matchAndRewritePermutation(op, rewriter, op.rhs(),
                                                        op.lhs())));
  }

  LogicalResult matchAndRewritePermutation(AddOp op, PatternRewriter &rewriter,
                                           Value thisOperand,
                                           Value otherOperand) const {
    ContractionOp contractOp =
        dyn_cast_or_null<ContractionOp>(thisOperand.getDefiningOp());
    if (!contractOp)
      return failure();

    if (!contractOp->hasOneUse())
      return failure();

    // Prevent possible cyclic uses; only allow BlockArguments or constants.
    if (otherOperand.getDefiningOp() &&
        !matchPattern(otherOperand, m_Constant()))
      return failure();

    // Prevent issues with broadcasts.
    if (op.result().getType() != contractOp.result().getType())
      return failure();

    // Add is the only legal aggregation kind for this pattern.
    if (contractOp.agg() != AggregationKind::add)
      return failure();

    FloatAttr init;
    if (!matchPattern(contractOp.init(), m_Constant(&init)))
      return failure();
    if (init.getValueAsDouble() != 0.0)
      return failure();

    contractOp.setOperand(0, otherOperand);
    rewriter.replaceOp(op, contractOp.result());

    return success();
  }
};

struct AlgebraicOptPass : public AlgebraicOptBase<AlgebraicOptPass> {
  void runOnOperation() final {
    FuncOp op = getOperation();
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
