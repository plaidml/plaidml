// Copyright 2020 Intel Corporation

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
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

// Change new added SCF to standards and avoid influence on gpu::LaunchOp
// content
struct ForLowering : public OpRewritePattern<mlir::scf::ForOp> {
  using OpRewritePattern<mlir::scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::scf::ForOp forOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace

/// Changes loadOp from i1 memref to loadOp i32 followed by creting constant
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

// From LowerToCFGPass, avoid conversion in gpu::LaunchOp contents.
LogicalResult ForLowering::matchAndRewrite(scf::ForOp forOp,
                                           PatternRewriter &rewriter) const {
  auto parentOp = forOp.getParentOp();
  if (isa<gpu::LaunchOp>(*parentOp)) {
    // Avoid conversion in gpu::LaunchOp region
    return failure();
  } else if (isa<scf::ForOp>(*parentOp)) {
    // Avoid conversion jump into nested scf loops
    return failure();
  }

  Location loc = forOp.getLoc();

  // Start by splitting the block containing the 'scf.for' into two parts.
  // The part before will get the init code, the part after will be the end
  // point.
  auto *initBlock = rewriter.getInsertionBlock();
  auto initPosition = rewriter.getInsertionPoint();
  auto *endBlock = rewriter.splitBlock(initBlock, initPosition);

  // Use the first block of the loop body as the condition block since it is the
  // block that has the induction variable and loop-carried values as arguments.
  // Split out all operations from the first block into a new block. Move all
  // body blocks from the loop body region to the region containing the loop.
  auto *conditionBlock = &forOp.region().front();
  auto *firstBodyBlock =
      rewriter.splitBlock(conditionBlock, conditionBlock->begin());
  auto *lastBodyBlock = &forOp.region().back();
  rewriter.inlineRegionBefore(forOp.region(), endBlock);
  auto iv = conditionBlock->getArgument(0);

  // Append the induction variable stepping logic to the last body block and
  // branch back to the condition block. Loop-carried values are taken from
  // operands of the loop terminator.
  Operation *terminator = lastBodyBlock->getTerminator();
  rewriter.setInsertionPointToEnd(lastBodyBlock);
  auto step = forOp.step();
  auto stepped = rewriter.create<AddIOp>(loc, iv, step).getResult();
  if (!stepped)
    return failure();

  SmallVector<Value, 8> loopCarried;
  loopCarried.push_back(stepped);
  loopCarried.append(terminator->operand_begin(), terminator->operand_end());
  rewriter.create<BranchOp>(loc, conditionBlock, loopCarried);
  rewriter.eraseOp(terminator);

  // Compute loop bounds before branching to the condition.
  rewriter.setInsertionPointToEnd(initBlock);
  Value lowerBound = forOp.lowerBound();
  Value upperBound = forOp.upperBound();
  if (!lowerBound || !upperBound)
    return failure();

  // The initial values of loop-carried values is obtained from the operands
  // of the loop operation.
  SmallVector<Value, 8> destOperands;
  destOperands.push_back(lowerBound);
  auto iterOperands = forOp.getIterOperands();
  destOperands.append(iterOperands.begin(), iterOperands.end());
  rewriter.create<BranchOp>(loc, conditionBlock, destOperands);

  // With the body block done, we can fill in the condition block.
  rewriter.setInsertionPointToEnd(conditionBlock);
  auto comparison =
      rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, iv, upperBound);

  rewriter.create<CondBranchOp>(loc, comparison, firstBodyBlock,
                                ArrayRef<Value>(), endBlock, ArrayRef<Value>());
  // The result of the loop operation is the values of the condition block
  // arguments except the induction variable on the last iteration.
  rewriter.replaceOp(forOp, conditionBlock->getArguments().drop_front());
  return success();
}

/// Hook for adding patterns.
void populateI1StorageToI32(MLIRContext *context,
                            OwningRewritePatternList &patterns) {
  patterns.insert<LoadOpI1ToI32, StoreOpI1ToI32, ForLowering>(context);
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

          Block &endBlock = function.back();
          auto const0 =
              builder
                  .create<ConstantOp>(loc, builder.getIntegerAttr(intType, 0))
                  .getResult();
          auto const1 =
              builder
                  .create<ConstantOp>(loc, builder.getIntegerAttr(intType, 1))
                  .getResult();

          // Use EDSC in ScopedContext
          {
            mlir::edsc::ScopedContext context(builder, loc);
            mlir::edsc::MemRefBoundsCapture mBoundsCapture(argument);

            SmallVector<Value, 8> steps;
            int count = memRefType.getRank();
            steps.reserve(count);
            auto index1 =
                builder.create<ConstantOp>(loc, builder.getIndexAttr(1));
            for (; count > 0; count--)
              steps.push_back(index1);

            // Build the scf loop to copy form i1 memref to i32 memref
            mlir::scf::buildLoopNest(
                builder, loc, mBoundsCapture.getLbs(), mBoundsCapture.getUbs(),
                steps,
                [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
                  auto valueToStore =
                      nestedBuilder.create<LoadOp>(loc, argument, ivs);
                  auto selOp = nestedBuilder.create<SelectOp>(loc, valueToStore,
                                                              const1, const0);
                  nestedBuilder.create<StoreOp>(loc, selOp.getResult(), alloc,
                                                ivs);
                });

            // Make sure to copy in reverse at the end of the block
            builder.setInsertionPoint(endBlock.getTerminator());

            // Build the scf loop to copy form i32 memref to i1 memref
            mlir::scf::buildLoopNest(
                builder, loc, mBoundsCapture.getLbs(), mBoundsCapture.getUbs(),
                steps,
                [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
                  auto valueToStore =
                      nestedBuilder.create<LoadOp>(loc, alloc, ivs);
                  auto cmpOp = nestedBuilder.create<CmpIOp>(
                      loc, CmpIPredicate::ne, valueToStore, const0);
                  nestedBuilder.create<StoreOp>(loc, cmpOp.getResult(),
                                                argument, ivs);
                });
          }

          // Make sure to deallocate this alloc at the end of the block.
          builder.setInsertionPoint(endBlock.getTerminator());
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
