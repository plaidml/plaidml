// Copyright 2021, Intel Corporation

#include "pmlc/conversion/linalg_to_pxa/pass_detail.h"

namespace pmlc::conversion::linalg_to_pxa {

using namespace mlir;         // NOLINT
using namespace mlir::linalg; // NOLINT

void buildSimpleYieldBody(OpBuilder &builder, Location loc, unsigned numInputs,
                          ValueRange args) {
  assert(args.size() == 2);
  builder.create<linalg::YieldOp>(loc, args[0]);
}

// TensorCollapseShapeOp does not have proper iterator_types() and
// getLoopsToShapesMap()
struct GeneralizeTensorCollapseShapeOp
    : public OpRewritePattern<TensorCollapseShapeOp> {
  using OpRewritePattern<TensorCollapseShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorCollapseShapeOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = op.src().getType().cast<ShapedType>();
    auto srcShape = srcType.getShape();
    MLIRContext *context = op.getContext();
    SmallVector<AffineExpr, 4> exprs;
    int numIdxs = 0;

    // Prepare for the src and dst maps
    for (auto dimAttr : op.reassociation()) {
      auto dims = dimAttr.cast<ArrayAttr>().getValue();
      AffineExpr expr = getAffineConstantExpr(0, context);
      int64_t stride = 1;
      for (int i = numIdxs + dims.size() - 1; i >= numIdxs; --i) {
        assert(dims[i - numIdxs].cast<IntegerAttr>().getInt() == i &&
               "Reassociation is not in order.");
        expr = getAffineConstantExpr(stride, context) *
                   getAffineDimExpr(i, context) +
               expr;
        stride *= srcShape[i];
      }
      numIdxs += dims.size();
      exprs.emplace_back(expr);
    }

    AffineMap inputMap = AffineMap::getMultiDimIdentityMap(numIdxs, context);
    AffineMap outputMap = AffineMap::get(numIdxs, 0, exprs, context);

    auto genericOp = createGenericOp(
        /*builder=*/rewriter,
        /*locationOp=*/op,
        /*outputTypes=*/TypeRange{op.getResult().getType().cast<ShapedType>()},
        /*inputs=*/ValueRange{op.src()},
        /*outputs=*/ValueRange{},
        /*numIdxs=*/numIdxs,
        /*maps=*/ArrayRef<AffineMap>{inputMap, outputMap},
        /*iterTypes=*/SmallVector<StringRef, 4>(numIdxs, "parallel"),
        /*bodyBuilder=*/buildSimpleYieldBody);
    op.getResult().replaceAllUsesWith(genericOp.getResult(0));
    rewriter.eraseOp(op);

    return success();
  }
};

void populateLinalgTensorCollapseOpGeneralizationPatterns(
    mlir::RewritePatternSet &patterns) {
  patterns.add<GeneralizeTensorCollapseShapeOp>(patterns.getContext());
}

// TensorExpandShapeOp does not have proper iterator_types() and
// getLoopsToShapesMap()
struct GeneralizeTensorExpandShapeOp
    : public OpRewritePattern<TensorExpandShapeOp> {
  using OpRewritePattern<TensorExpandShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorExpandShapeOp op,
                                PatternRewriter &rewriter) const override {
    auto dstType = op.result().getType().cast<ShapedType>();
    auto dstShape = dstType.getShape();
    MLIRContext *context = op.getContext();
    SmallVector<AffineExpr, 4> exprs;
    int numIdxs = 0;

    // Prepare for the src and dst maps
    for (auto dimAttr : op.reassociation()) {
      auto dims = dimAttr.cast<ArrayAttr>().getValue();
      AffineExpr expr = getAffineConstantExpr(0, context);
      int64_t stride = 1;
      for (int i = numIdxs + dims.size() - 1; i >= numIdxs; --i) {
        assert(dims[i - numIdxs].cast<IntegerAttr>().getInt() == i &&
               "Reassociation is not in order.");
        expr = getAffineConstantExpr(stride, context) *
                   getAffineDimExpr(i, context) +
               expr;
        stride *= dstShape[i];
      }
      numIdxs += dims.size();
      exprs.emplace_back(expr);
    }

    AffineMap inputMap = AffineMap::get(numIdxs, 0, exprs, context);
    AffineMap outputMap = AffineMap::getMultiDimIdentityMap(numIdxs, context);

    auto genericOp = createGenericOp(
        /*builder=*/rewriter,
        /*locationOp=*/op,
        /*outputTypes=*/TypeRange{op.getResult().getType().cast<ShapedType>()},
        /*inputs=*/ValueRange{op.src()},
        /*outputs=*/ValueRange{},
        /*numIdxs=*/numIdxs,
        /*maps=*/ArrayRef<AffineMap>{inputMap, outputMap},
        /*iterTypes=*/SmallVector<StringRef, 4>(numIdxs, "parallel"),
        /*bodyBuilder=*/buildSimpleYieldBody);
    op.getResult().replaceAllUsesWith(genericOp.getResult(0));
    rewriter.eraseOp(op);

    return success();
  }
};

void populateLinalgTensorExpandOpGeneralizationPatterns(
    mlir::RewritePatternSet &patterns) {
  patterns.add<GeneralizeTensorExpandShapeOp>(patterns.getContext());
}

} // namespace pmlc::conversion::linalg_to_pxa
