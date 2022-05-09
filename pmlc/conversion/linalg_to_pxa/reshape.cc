// Copyright 2021, Intel Corporation

#include "pmlc/conversion/linalg_to_pxa/pass_detail.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace pmlc::conversion::linalg_to_pxa {

using namespace mlir; // NOLINT

void buildSimpleYieldBody(OpBuilder &builder, Location loc, unsigned numInputs,
                          ValueRange args) {
  assert(args.size() == 2);
  builder.create<linalg::YieldOp>(loc, args[0]);
}

// tensor::CollapseShapeOp does not have proper iterator_types() and
// getLoopsToShapesMap()
// TODO: Lorenzo: check if this is still true.
struct GeneralizeTensorCollapseShapeOp
    : public OpRewritePattern<tensor::CollapseShapeOp> {
  using OpRewritePattern<tensor::CollapseShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::CollapseShapeOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = op.src().getType().cast<ShapedType>();
    auto srcShape = srcType.getShape();
    MLIRContext *context = op.getContext();
    int numIdxs = srcShape.size();
    AffineMap inputMap = AffineMap::getMultiDimIdentityMap(numIdxs, context);
    AffineMap outputMap;

    if (op.reassociation().empty()) {
      outputMap = AffineMap::get(numIdxs, 0, {}, context);
    } else {
      SmallVector<AffineExpr, 4> exprs;
      int curIdxs = 0;
      // Prepare for the src and dst maps
      for (auto dimAttr : op.reassociation()) {
        auto dims = dimAttr.cast<ArrayAttr>().getValue();
        AffineExpr expr = getAffineConstantExpr(0, context);
        int64_t stride = 1;
        for (int i = curIdxs + dims.size() - 1; i >= curIdxs; --i) {
          assert(dims[i - curIdxs].cast<IntegerAttr>().getInt() == i &&
                 "Reassociation is not in order.");
          expr = getAffineConstantExpr(stride, context) *
                     getAffineDimExpr(i, context) +
                 expr;
          stride *= srcShape[i];
        }
        curIdxs += dims.size();
        exprs.emplace_back(expr);
      }
      assert(numIdxs == curIdxs);
      outputMap = AffineMap::get(numIdxs, 0, exprs, context);
    }

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
    RewritePatternSet &patterns) {
  patterns.add<GeneralizeTensorCollapseShapeOp>(patterns.getContext());
}

// tensor::ExpandShapeOp does not have proper iterator_types() and
// getLoopsToShapesMap()
struct GeneralizeTensorExpandShapeOp
    : public OpRewritePattern<tensor::ExpandShapeOp> {
  using OpRewritePattern<tensor::ExpandShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp op,
                                PatternRewriter &rewriter) const override {
    auto dstType = op.result().getType().cast<ShapedType>();
    auto dstShape = dstType.getShape();
    MLIRContext *context = op.getContext();
    int numIdxs = dstShape.size();
    AffineMap inputMap;
    AffineMap outputMap = AffineMap::getMultiDimIdentityMap(numIdxs, context);

    if (op.reassociation().empty()) {
      inputMap = AffineMap::get(numIdxs, 0, {}, context);
    } else {
      SmallVector<AffineExpr, 4> exprs;
      int curIdxs = 0;
      // Prepare for the src and dst maps
      for (auto dimAttr : op.reassociation()) {
        auto dims = dimAttr.cast<ArrayAttr>().getValue();
        AffineExpr expr = getAffineConstantExpr(0, context);
        int64_t stride = 1;
        for (int i = curIdxs + dims.size() - 1; i >= curIdxs; --i) {
          assert(dims[i - curIdxs].cast<IntegerAttr>().getInt() == i &&
                 "Reassociation is not in order.");
          expr = getAffineConstantExpr(stride, context) *
                     getAffineDimExpr(i, context) +
                 expr;
          stride *= dstShape[i];
        }
        curIdxs += dims.size();
        exprs.emplace_back(expr);
      }
      assert(numIdxs == curIdxs);
      inputMap = AffineMap::get(numIdxs, 0, exprs, context);
    }

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
    RewritePatternSet &patterns) {
  patterns.add<GeneralizeTensorExpandShapeOp>(patterns.getContext());
}

} // namespace pmlc::conversion::linalg_to_pxa
