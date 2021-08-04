// Copyright 2021, Intel Corporation

#include "pmlc/conversion/linalg_to_pxa/pass_detail.h"

namespace pmlc::conversion::linalg_to_pxa {

using namespace mlir;         // NOLINT
using namespace mlir::linalg; // NOLINT

void buildPoolingMaxOpBody(OpBuilder &builder, Location loc, unsigned numInputs,
                           ValueRange args) {
  Value cmpResult;
  if (args[2].getType().isa<IntegerType>()) {
    cmpResult =
        builder.create<CmpIOp>(loc, CmpIPredicate::sgt, args[2], args[0])
            .getResult();
  } else if (args[2].getType().isa<FloatType>()) {
    cmpResult =
        builder.create<CmpFOp>(loc, CmpFPredicate::OGT, args[2], args[0])
            .getResult();
  } else {
    builder.getBlock()->getParentOp()->emitError(
        "The input value is not integer or float for pooling max.");
  }
  auto result = builder.create<SelectOp>(loc, cmpResult, args[2], args[0]);
  builder.create<linalg::YieldOp>(builder.loc, result.getResult());
}

void buildPoolingMinOpBody(OpBuilder &builder, Location loc, unsigned numInputs,
                           ValueRange args) {
  Value cmpResult;
  if (args[2].getType().isa<IntegerType>()) {
    cmpResult =
        builder.create<CmpIOp>(loc, CmpIPredicate::slt, args[2], args[0])
            .getResult();
  } else if (args[2].getType().isa<FloatType>()) {
    cmpResult =
        builder.create<CmpFOp>(loc, CmpFPredicate::OLT, args[2], args[0])
            .getResult();
  } else {
    builder.getBlock()->getParentOp()->emitError(
        "The input value is not integer or float for pooling min.");
  }
  auto result = builder.create<SelectOp>(loc, cmpResult, args[2], args[0]);
  builder.create<linalg::YieldOp>(loc, result.getResult());
}

void buildPoolingSumOpBody(OpBuilder &builder, Location loc, unsigned numInputs,
                           ValueRange args) {
  Value result;
  if (args[2].getType().isa<IntegerType>()) {
    result = builder.create<AddIOp>(loc, args[2], args[0]).getResult();
  } else if (args[2].getType().isa<FloatType>()) {
    result = builder.create<AddFOp>(loc, args[2], args[0]).getResult();
  } else {
    builder.getBlock()->getParentOp()->emitError(
        "The input value is not integer or float for pooling sum.");
  }
  builder.create<linalg::YieldOp>(loc, result);
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
    // Window's affine map
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

    SmallVector<StringRef, 4> iterTypes(
        op.iterator_types().template getAsValueRange<StringAttr>());
    auto genericOp = createGenericOp(
        /*builder=*/rewriter,
        /*locationOp=*/op,
        /*outputTypes=*/TypeRange{outputType},
        /*inputs=*/ValueRange{op.input(), op.windowDims()},
        /*outputs=*/ValueRange{op.output()},
        /*numIdxs=*/numIdxs,
        /*maps=*/ArrayRef<AffineMap>{inputMap, windowMap, outputMap},
        /*iterTypes*/ iterTypes,
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
