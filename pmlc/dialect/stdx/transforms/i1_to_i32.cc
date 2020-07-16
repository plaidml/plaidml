// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "pmlc/dialect/stdx/transforms/pass_detail.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::stdx {

namespace {

/// Changes loadOp from i1 memref to loadOp i32 followed by creting constant
/// value 0 of i32 type and doing cmpi afterwards, next this bool-like type will
/// be produced
class LoadOpI1ToI32 final : public OpRewritePattern<LoadOp> {
public:
  using OpRewritePattern<LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoadOp loadOp,
                                PatternRewriter &rewriter) const override;
};

/// Changes storeOp to i1 memref to i32 select(val, 1, 0) followed by storeOp to
/// i32
class StoreOpI1ToI32 final : public OpRewritePattern<StoreOp> {
public:
  using OpRewritePattern<StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StoreOp storeOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace

/// Changes loadOp from i1 memref to loadOp i32 followed by creting constant
/// value 0 of i32 type and doing cmpi afterwards, next this bool-like type will
/// be produced
LogicalResult LoadOpI1ToI32::matchAndRewrite(LoadOp loadOp,
                                             PatternRewriter &rewriter) const {
  auto elementType = loadOp.result().getType();
  if (!elementType.isInteger(1)) {
    return failure();
  }

  auto destType = rewriter.getIntegerType(32);
  auto loc = loadOp.getLoc();

  loadOp.getMemRef().setType(
      MemRefType::get(loadOp.getMemRefType().getShape(), destType));

  auto newLoadOp =
      rewriter.create<LoadOp>(loc, loadOp.memref(), loadOp.indices());

  auto const0 =
      rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(destType, 0));
  auto cmpOp = rewriter.create<CmpIOp>(
      loc, CmpIPredicate::ne, newLoadOp.getResult(), const0.getResult());

  rewriter.replaceOp(loadOp, {cmpOp});
  return success();
}

/// Changes storeOp to i1 memref to i32 select(val, 1, 0) followed by storeOp to
/// i32
LogicalResult StoreOpI1ToI32::matchAndRewrite(StoreOp storeOp,
                                              PatternRewriter &rewriter) const {
  auto elementType = storeOp.value().getType();
  if (!elementType.isInteger(1)) {
    return failure();
  }

  auto destType = rewriter.getIntegerType(32);
  auto loc = storeOp.getLoc();

  auto const0 =
      rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(destType, 0));
  auto const1 =
      rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(destType, 1));
  auto selOp = rewriter.create<SelectOp>(
      loc, storeOp.value(), const1.getResult(), const0.getResult());

  storeOp.getMemRef().setType(
      MemRefType::get(storeOp.getMemRefType().getShape(), destType));

  rewriter.replaceOpWithNewOp<StoreOp>(storeOp, selOp.getResult(),
                                       storeOp.memref(), storeOp.indices());

  return success();
}

/// Hook for adding patterns.
void populateI1StorageToI32(MLIRContext *context,
                            OwningRewritePatternList &patterns) {
  patterns.insert<LoadOpI1ToI32, StoreOpI1ToI32>(context);
}

struct I1StorageToI32Pass : public I1StorageToI32Base<I1StorageToI32Pass> {
  void runOnFunction() final {
    OwningRewritePatternList patterns;
    auto *context = &getContext();

    auto function = getFunction();
    Block &entryBlock = function.front();
    for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; i++)
      if (auto memRefType = entryBlock.getArgument(i)
                                .getType()
                                .dyn_cast_or_null<MemRefType>()) {
        if (memRefType.getElementType().isInteger(1)) {
          // shall convert this type to int32
          auto argument = entryBlock.getArgument(i);
          auto intType = IntegerType::get(32, context);
          auto newMemRefType = MemRefType::get(memRefType.getShape(), intType);
          OpBuilder builder(context);
          Location loc = entryBlock.front().getLoc();

          builder.setInsertionPointToStart(&entryBlock);
          auto alloc = builder.create<AllocOp>(loc, newMemRefType);

          // Make sure to allocate at the beginning of the block.
          auto *parentBlock = alloc.getOperation()->getBlock();
          alloc.getOperation()->moveBefore(&parentBlock->front());

          // Update old memref uses with new memref
          argument.replaceUsesWithIf(alloc,
                                     [&](OpOperand &operand) { return true; });

          auto const0 =
              builder
                  .create<ConstantOp>(loc, builder.getIntegerAttr(intType, 0))
                  .getResult();
          auto const1 =
              builder
                  .create<ConstantOp>(loc, builder.getIntegerAttr(intType, 1))
                  .getResult();
          SmallVector<int64_t, 4> lowerBounds(memRefType.getRank(),
                                              /*Value=*/0);
          SmallVector<int64_t, 4> steps(memRefType.getRank(), /*Value=*/1);

          // Copy data from memref<i1> to memref<i32>
          buildAffineLoopNest(
              builder, loc, lowerBounds, memRefType.getShape(), steps,
              [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
                auto valueToStore =
                    nestedBuilder.create<AffineLoadOp>(loc, argument, ivs);
                auto selOp = nestedBuilder.create<SelectOp>(loc, valueToStore,
                                                            const1, const0);
                nestedBuilder.create<AffineStoreOp>(loc, selOp.getResult(),
                                                    alloc, ivs);
              });

          // Make sure to allocate at the end of the block.
          // Copy data from memref<i32> to memref<i1>
          builder.setInsertionPoint(entryBlock.getTerminator());
          buildAffineLoopNest(
              builder, loc, lowerBounds, memRefType.getShape(), steps,
              [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
                auto valueToStore =
                    nestedBuilder.create<AffineLoadOp>(loc, alloc, ivs);
                auto cmpOp = nestedBuilder.create<CmpIOp>(
                    loc, CmpIPredicate::ne, valueToStore, const0);
                nestedBuilder.create<AffineStoreOp>(loc, cmpOp.getResult(),
                                                    argument, ivs);
              });

          // Make sure to deallocate this alloc at the end of the block.
          builder.create<DeallocOp>(loc, alloc);
        }
      }

    populateI1StorageToI32(context, patterns);
    applyPatternsAndFoldGreedily(getOperation(), patterns);
  }
};

std::unique_ptr<mlir::Pass> createI1StorageToI32Pass() {
  return std::make_unique<I1StorageToI32Pass>();
}

} // namespace pmlc::dialect::stdx
