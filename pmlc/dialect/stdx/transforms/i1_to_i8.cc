// Copyright 2020 Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "pmlc/dialect/stdx/transforms/pass_detail.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::stdx {

namespace {

/// Changes loadOp from i1 memref to loadOp i8 followed by trunci i8->i1
class LoadOpI1ToI8 final : public OpRewritePattern<LoadOp> {
public:
  using OpRewritePattern<LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoadOp loadOp,
                                PatternRewriter &rewriter) const override;
};

/// Changes storeOp to i1 memref to zexti i1->i8 followed by storeOp to i8
class StoreOpI1ToI8 final : public OpRewritePattern<StoreOp> {
public:
  using OpRewritePattern<StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StoreOp storeOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace

/// Changes loadOp from i1 memref to loadOp i8 followed by trunci i8->i1
LogicalResult LoadOpI1ToI8::matchAndRewrite(LoadOp loadOp,
                                            PatternRewriter &rewriter) const {
  auto elementType = loadOp.result().getType();
  if (!elementType.isInteger(1)) {
    return failure();
  }

  auto destType = rewriter.getIntegerType(8);
  auto loc = loadOp.getLoc();

  loadOp.getMemRef().setType(
      MemRefType::get(loadOp.getMemRefType().getShape(), destType));

  auto newLoadOp =
      rewriter.create<LoadOp>(loc, loadOp.memref(), loadOp.indices());
  auto truncOp = rewriter.create<TruncateIOp>(loc, newLoadOp.getResult(),
                                              rewriter.getIntegerType(1));

  rewriter.replaceOp(loadOp, {truncOp});
  return success();
}

/// Changes storeOp to i1 memref to zexti i1->i8 followed by storeOp to i8
LogicalResult StoreOpI1ToI8::matchAndRewrite(StoreOp storeOp,
                                             PatternRewriter &rewriter) const {
  auto elementType = storeOp.value().getType();
  if (!elementType.isInteger(1)) {
    return failure();
  }

  auto destType = rewriter.getIntegerType(8);
  auto loc = storeOp.getLoc();
  auto extOp = rewriter.create<ZeroExtendIOp>(loc, storeOp.value(), destType);

  storeOp.getMemRef().setType(
      MemRefType::get(storeOp.getMemRefType().getShape(), destType));

  rewriter.replaceOpWithNewOp<StoreOp>(storeOp, extOp.getResult(),
                                       storeOp.memref(), storeOp.indices());

  return success();
}

/// Hook for adding patterns.
void populateI1StorageToI8(MLIRContext *context,
                           OwningRewritePatternList &patterns) {
  patterns.insert<LoadOpI1ToI8, StoreOpI1ToI8>(context);
}

struct I1StorageToI8Pass : public I1StorageToI8Base<I1StorageToI8Pass> {
  void runOnFunction() final {
    OwningRewritePatternList patterns;
    auto *context = &getContext();

    populateI1StorageToI8(context, patterns);
    applyPatternsAndFoldGreedily(getOperation(), patterns);
  }
};

std::unique_ptr<mlir::Pass> createI1StorageToI8Pass() {
  return std::make_unique<I1StorageToI8Pass>();
}

} // namespace pmlc::dialect::stdx
