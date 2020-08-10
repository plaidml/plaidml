//===- SCFToGPUPass.cpp - Convert a loop nest to a GPU kernel -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pmlc/conversion/SCFToGPU/SCFToGPUPass.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"

#include "pmlc/conversion/SCFToGPU/PassDetail.h"
#include "pmlc/conversion/SCFToGPU/SCFToGPU.h"

#define PASS_NAME "convert-scf-to-gpu"
#define LOOPOP_TO_GPU_PASS_NAME "convert-loop-op-to-gpu"

using namespace mlir;      // NOLINT
using namespace mlir::scf; // NOLINT

namespace pmlc::conversion::scf_to_gpu {

namespace {
// Change new added SCF to standards and avoid influence on gpu::LaunchOp
// content
struct ForLowering : public OpRewritePattern<mlir::scf::ForOp> {
  using OpRewritePattern<mlir::scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::scf::ForOp forOp,
                                PatternRewriter &rewriter) const override;
};

// From LowerToCFGPass, avoid conversion in gpu::LaunchOp contents.
LogicalResult ForLowering::matchAndRewrite(scf::ForOp forOp,
                                           PatternRewriter &rewriter) const {
  Operation *nested = &forOp.getBody()->front();
  auto parentOp = forOp.getParentOp();
  if (dyn_cast<scf::ForOp>(nested) || isa<scf::ForOp>(*parentOp)) {
    // only single op changes
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
// A pass that traverses top-level loops in the function and converts them to
// GPU launch operations.  Nested launches are not allowed, so this does not
// walk the function recursively to avoid considering nested loops.
struct ForLoopMapper : public ConvertSimpleSCFToGPUBase<ForLoopMapper> {
  ForLoopMapper() = default;
  ForLoopMapper(unsigned numBlockDims, unsigned numThreadDims) {
    this->numBlockDims = numBlockDims;
    this->numThreadDims = numThreadDims;
  }

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<ForLowering>(context);
    applyPatternsAndFoldGreedily(getOperation(), patterns);

    for (Operation &op : llvm::make_early_inc_range(getFunction().getOps())) {
      if (auto forOp = dyn_cast<AffineForOp>(&op)) {
        if (failed(convertAffineLoopNestToGPULaunch(forOp, numBlockDims,
                                                    numThreadDims)))
          signalPassFailure();
      } else if (auto forOp = dyn_cast<ForOp>(&op)) {
        if (failed(
                convertLoopNestToGPULaunch(forOp, numBlockDims, numThreadDims)))
          signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
createSimpleSCFToGPUPass(unsigned numBlockDims, unsigned numThreadDims) {
  return std::make_unique<ForLoopMapper>(numBlockDims, numThreadDims);
}

std::unique_ptr<OperationPass<FuncOp>> createSimpleSCFToGPUPass() {
  return std::make_unique<ForLoopMapper>();
}

} // namespace pmlc::conversion::scf_to_gpu
