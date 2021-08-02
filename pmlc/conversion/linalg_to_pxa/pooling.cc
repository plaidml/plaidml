// Copyright 2021, Intel Corporation

#include "pmlc/conversion/linalg_to_pxa/pass_detail.h"

namespace pmlc::conversion::linalg_to_pxa {

using namespace mlir;         // NOLINT
using namespace mlir::linalg; // NOLINT

void buildPoolingMaxOpBody(OpBuilder &builder, unsigned numInputs,
                           ValueRange args) {
  Value cmpResult;
  if (args[2].getType().isa<IntegerType>()) {
    cmpResult = builder
                    .create<CmpIOp>(builder.getUnknownLoc(), CmpIPredicate::sgt,
                                    args[2], args[0])
                    .getResult();
  } else if (args[2].getType().isa<FloatType>()) {
    cmpResult = builder
                    .create<CmpFOp>(builder.getUnknownLoc(), CmpFPredicate::OGT,
                                    args[2], args[0])
                    .getResult();
  } else {
    builder.getBlock()->getParentOp()->emitError(
        "The input value is not integer or float for pooling sum.");
  }
  auto result = builder.create<SelectOp>(builder.getUnknownLoc(), cmpResult,
                                         args[2], args[0]);
  builder.create<linalg::YieldOp>(builder.getUnknownLoc(), result.getResult());
}

void buildPoolingMinOpBody(OpBuilder &builder, unsigned numInputs,
                           ValueRange args) {
  Value cmpResult;
  if (args[2].getType().isa<IntegerType>()) {
    cmpResult = builder
                    .create<CmpIOp>(builder.getUnknownLoc(), CmpIPredicate::slt,
                                    args[2], args[0])
                    .getResult();
  } else if (args[2].getType().isa<FloatType>()) {
    cmpResult = builder
                    .create<CmpFOp>(builder.getUnknownLoc(), CmpFPredicate::OLT,
                                    args[2], args[0])
                    .getResult();
  } else {
    builder.getBlock()->getParentOp()->emitError(
        "The input value is not integer or float for pooling sum.");
  }
  auto result = builder.create<SelectOp>(builder.getUnknownLoc(), cmpResult,
                                         args[2], args[0]);
  builder.create<linalg::YieldOp>(builder.getUnknownLoc(), result.getResult());
}

void buildPoolingSumOpBody(OpBuilder &builder, unsigned numInputs,
                           ValueRange args) {
  Value result;
  if (args[2].getType().isa<IntegerType>()) {
    result = builder.create<AddIOp>(builder.getUnknownLoc(), args[2], args[0])
                 .getResult();
  } else if (args[2].getType().isa<FloatType>()) {
    result = builder.create<AddFOp>(builder.getUnknownLoc(), args[2], args[0])
                 .getResult();
  } else {
    builder.getBlock()->getParentOp()->emitError(
        "The input value is not integer or float for pooling sum.");
  }
  builder.create<linalg::YieldOp>(builder.getUnknownLoc(), result);
}

template <typename PoolingOpType>
struct GeneralizePoolingOp : public OpRewritePattern<PoolingOpType> {
  using OpRewritePattern<PoolingOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(PoolingOpType op,
                                PatternRewriter &rewriter) const override {
    AffineMap shapesMap = op.getLoopsToShapesMap();

    Type origType = op.output().getType();
    ShapedType outputType = origType.cast<ShapedType>();
    unsigned numIdxs = shapesMap.getNumDims();
    auto exprs = shapesMap.getResults();
    MLIRContext *context = op.getContext();
    // Input's affine map
    Type inputType = op.input().getType();
    unsigned numInputDims = inputType.cast<ShapedType>().getRank();
    SmallVector<AffineExpr, 4> inputExprs(exprs.begin(),
                                          exprs.begin() + numInputDims);
    AffineMap inputMap = AffineMap::get(numIdxs, 0, inputExprs, context);
    // Input's affine map
    Type windowType = op.windowDims().getType();
    unsigned numWindowDims = windowType.cast<ShapedType>().getRank();
    SmallVector<AffineExpr, 4> windowExprs(exprs.begin() + numInputDims,
                                           exprs.begin() +
                                               (numInputDims + numWindowDims));
    AffineMap windowMap = AffineMap::get(numIdxs, 0, windowExprs, context);
    // Output's affine map
    SmallVector<AffineExpr, 4> outputExprs(
        exprs.begin() + (numInputDims + numWindowDims), exprs.end());
    AffineMap outputMap = AffineMap::get(numIdxs, 0, outputExprs, context);

    // Use different body builder for different pooling ops
    GenericOpBodyBuilder bodyBuilder;
    if (isa<PoolingMaxOp>(op)) {
      bodyBuilder = buildPoolingMaxOpBody;
    } else if (isa<PoolingMinOp>(op)) {
      bodyBuilder = buildPoolingMinOpBody;
    } else if (isa<PoolingSumOp>(op)) {
      bodyBuilder = buildPoolingSumOpBody;
    } else {
      op.emitError("Invalid pooling op.");
    }

    auto genericOp = createGenericOp(
        /*builder=*/rewriter,
        /*locationOp=*/op,
        /*outputTypes=*/TypeRange{outputType},
        /*inputs=*/ValueRange{op.input(), op.windowDims()},
        /*outputs=*/ValueRange{op.output()},
        /*numIdxs=*/numIdxs,
        /*maps=*/ArrayRef<AffineMap>{inputMap, windowMap, outputMap},
        /*bodyBuilder=*/bodyBuilder);

    // Do not replace the use of the output in the new generic op
    op.output().replaceUsesWithIf(genericOp.getResult(0), [&](OpOperand &use) {
      return genericOp.getOperation() != use.getOwner();
    });
    rewriter.eraseOp(op);
    return success();
  }
};

void populateLinalgPoolingOpGeneralizationPatterns(
    mlir::RewritePatternSet &patterns) {
  patterns.add<GeneralizePoolingOp<PoolingMaxOp>, //
               GeneralizePoolingOp<PoolingMinOp>, //
               GeneralizePoolingOp<PoolingSumOp>>(patterns.getContext());
}

} // namespace pmlc::conversion::linalg_to_pxa
