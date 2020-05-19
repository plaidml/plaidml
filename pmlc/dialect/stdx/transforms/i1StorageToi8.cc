// Copyright 2020 Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::stdx {

namespace {

/// Changes loadOp from i1 memref to loadOp i8 followed by trunci i8->i1
class LoadOpi1Toi8 final : public OpRewritePattern<LoadOp> {
public:
  using OpRewritePattern<LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoadOp loadOp,
                                PatternRewriter &rewriter) const override;
};

/// Changes storeOp to i1 memref to zexti i1->i8 followed by storeOp to i8
class StoreOpi1Toi8 final : public OpRewritePattern<StoreOp> {
public:
  using OpRewritePattern<StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StoreOp storeOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace

/// Changes loadOp from i1 memref to loadOp i8 followed by trunci i8->i1
LogicalResult LoadOpi1Toi8::matchAndRewrite(LoadOp loadOp,
                                            PatternRewriter &rewriter) const {
  auto elementType = loadOp.result().getType();
  if (!elementType.isInteger(1)) {
    return success();
  }

  auto destType = rewriter.getIntegerType(8);
  auto loc = loadOp.getLoc();

  loadOp.getMemRef().setType(
      MemRefType::get(loadOp.getMemRefType().getShape(), destType));

  auto loadOp_new =
      rewriter.create<LoadOp>(loc, loadOp.memref(), loadOp.indices());
  auto truncOp = rewriter.create<TruncateIOp>(loc, loadOp_new.getResult(),
                                              rewriter.getIntegerType(1));

  rewriter.replaceOp(loadOp, {truncOp});
  return success();
}

/// Changes storeOp to i1 memref to zexti i1->i8 followed by storeOp to i8
LogicalResult StoreOpi1Toi8::matchAndRewrite(StoreOp storeOp,
                                             PatternRewriter &rewriter) const {
  auto elementType = storeOp.value().getType();
  if (!elementType.isInteger(1)) {
    return success();
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
void populatei1StorageToi8(MLIRContext *context,
                           OwningRewritePatternList &patterns) {
  patterns.insert<LoadOpi1Toi8, StoreOpi1Toi8>(context);
}

struct i1StorageToi8Pass
    : public mlir::PassWrapper<i1StorageToi8Pass, mlir::FunctionPass> {
  void runOnFunction() final {
    OwningRewritePatternList patterns;
    auto *context = &getContext();

    populatei1StorageToi8(context, patterns);
    applyPatternsAndFoldGreedily(getOperation(), patterns);
  }
};

std::unique_ptr<mlir::Pass> createi1StorageToi8Pass() {
  return std::make_unique<i1StorageToi8Pass>();
}

} // namespace pmlc::dialect::stdx
