// Copyright 2020 Intel Corporation

#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "pmlc/dialect/stdx/transforms/pass_detail.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::stdx {

namespace {

/// Changes loadOp from i1 memref to loadOp i32 followed by creating constant
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

/// Changes loadOp from i1 memref to loadOp i32 followed by creating constant
/// value 0 of i32 type and doing cmpi afterwards, next this bool-like type will
/// be produced
LogicalResult LoadOpI1ToI32::matchAndRewrite(LoadOp loadOp,
                                             PatternRewriter &rewriter) const {
  // Not convert function argument as we alloc i32 buffer for it.
  if (!loadOp.getMemRef().getDefiningOp()) {
    return failure();
  }

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
  // Not convert function argument as we alloc i32 buffer for it.
  if (!storeOp.getMemRef().getDefiningOp()) {
    return failure();
  }

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

// Adaptor for building loop nests for the specific memRef shape
void buildLoopForMemRef(
    OpBuilder &builder, Location loc, Value memRef,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder) {
  auto memRefType = memRef.getType().dyn_cast_or_null<MemRefType>();
  if (!memRefType)
    return;

  mlir::edsc::ScopedContext context(builder, loc);
  mlir::edsc::MemRefBoundsCapture mBoundsCapture(memRef);

  auto indexConst1 = builder.create<ConstantOp>(loc, builder.getIndexAttr(1));
  SmallVector<Value, 8> steps, ivs;
  int rank = memRefType.getRank();
  steps.reserve(rank);
  for (int i = 0; i < rank; i++)
    steps.push_back(indexConst1);

  Block *currentBlock = builder.getInsertionBlock();
  Block *ifBlock, *thenBlock, *continueBlock;

  for (int i = 0; i < rank; i++) {
    // Create blocks
    auto currentPoint = builder.getInsertionPoint();
    ifBlock = currentBlock->splitBlock(currentPoint);
    thenBlock = ifBlock->splitBlock(ifBlock->begin());
    continueBlock = thenBlock->splitBlock(thenBlock->begin());

    // Create loop entry
    builder.setInsertionPointToEnd(currentBlock);
    builder.create<BranchOp>(loc, ifBlock, mBoundsCapture.lb(i));

    // Check up bound
    builder.setInsertionPointToStart(ifBlock);
    ifBlock->addArgument(mBoundsCapture.lb(i).getType());
    auto comparsion = builder.create<CmpIOp>(
        loc, CmpIPredicate::slt, ifBlock->getArgument(0), mBoundsCapture.ub(i));
    builder.create<CondBranchOp>(loc, comparsion, thenBlock, ArrayRef<Value>(),
                                 continueBlock, ArrayRef<Value>());

    // Update index with step
    builder.setInsertionPointToStart(thenBlock);
    auto stepped =
        builder.create<AddIOp>(loc, ifBlock->getArgument(0), steps[i])
            .getResult();
    builder.create<BranchOp>(loc, ifBlock, stepped);

    // Ready for next dimension
    builder.setInsertionPointToStart(thenBlock);
    currentBlock = thenBlock;
    ivs.push_back(ifBlock->getArgument(0));
  }

  // Create loop body
  if (thenBlock && bodyBuilder) {
    bodyBuilder(builder, loc, ivs);
  }
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
          // shall convert this memref to int32
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

          // create mem copy from i1 buffer to i32 buffer
          buildLoopForMemRef(
              builder, loc, argument,
              [&](OpBuilder &builder, Location loc, ValueRange ivs) {
                auto valueToStore = builder.create<LoadOp>(loc, argument, ivs);
                auto selOp =
                    builder.create<SelectOp>(loc, valueToStore, const1, const0);
                builder.create<StoreOp>(loc, selOp.getResult(), alloc, ivs);
              });

          // Make sure to deallocate this alloc at the end of the block.
          Block &endBlock = function.back();
          builder.setInsertionPoint(endBlock.getTerminator());
          auto dealloc = builder.create<DeallocOp>(loc, alloc);

          // create mem copy from i32 buffer to i1 buffer
          builder.setInsertionPoint(dealloc);
          buildLoopForMemRef(
              builder, loc, argument,
              [&](OpBuilder &builder, Location loc, ValueRange ivs) {
                auto valueToStore = builder.create<LoadOp>(loc, alloc, ivs);
                auto cmpOp = builder.create<CmpIOp>(loc, CmpIPredicate::ne,
                                                    valueToStore, const0);
                builder.create<StoreOp>(loc, cmpOp.getResult(), argument, ivs);
              });
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
