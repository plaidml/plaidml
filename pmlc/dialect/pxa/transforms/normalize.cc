// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/normalize.h"

#include <memory>

#include "llvm/ADT/StringSet.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/PatternMatch.h"

#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/util/tags.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

/// Promotes the loop body of an affine.parallel to its containing block if no
/// induction variables are present.
void promoteIfEmptyIVs(AffineParallelOp op) {
  // Nothing to do when there are induction variables.
  if (op.getNumDims())
    return;

  // Don't remove any affine.parallel loop with tags
  if (hasTags(op))
    return;

  // Replace yielded loop results.
  auto *body = op.getBody();
  auto yield = cast<AffineYieldOp>(body->back());
  op.replaceAllUsesWith(yield.operands());

  // Move the loop body operations, except for its terminator, to the loop's
  // containing block.
  yield.erase();
  auto &parentOps = op.getOperation()->getBlock()->getOperations();
  parentOps.splice(Block::iterator(op), body->getOperations());
  op.erase();
}

void elideSingleIterationIndexes(AffineParallelOp op) {
  AffineValueMap ranges = util::getRangesValueMap(op);
  Block *body = op.getBody();
  SmallVector<AffineExpr, 6> newLowerBounds;
  SmallVector<AffineExpr, 6> newUpperBounds;
  SmallVector<int64_t, 6> newSteps;
  SmallVector<BlockArgument, 6> argsToRemove;
  SmallVector<int32_t> groups;
  auto steps = op.getSteps();
  for (unsigned i = 0, e = body->getNumArguments(); i < e; i++) {
    // Is the range a constant value matching the step size?
    auto constExpr = ranges.getResult(i).dyn_cast<AffineConstantExpr>();
    int64_t step = steps[i];
    if (constExpr && constExpr.getValue() == step) {
      // Mark argument for removal and replacement with 0.
      argsToRemove.push_back(body->getArgument(i));
    } else {
      // Keep argument
      newLowerBounds.push_back(op.lowerBoundsMap().getResult(i));
      newUpperBounds.push_back(op.upperBoundsMap().getResult(i));
      newSteps.push_back(step);
      groups.push_back(1);
    }
  }

  // Return if no arguments need removal.
  if (argsToRemove.empty())
    return;

  auto builder = OpBuilder::atBlockBegin(body);
  for (auto arg : argsToRemove) {
    auto argNumber = arg.getArgNumber();
    auto lowerBoundValue = builder.create<AffineApplyOp>(
        op.getLoc(), op.lowerBoundsMap().getSubMap({argNumber}),
        op.getLowerBoundsOperands());
    arg.replaceAllUsesWith(lowerBoundValue);
    body->eraseArgument(argNumber);
  }

  // Update attributes and return success
  auto newLower = AffineMap::get(op.lowerBoundsMap().getNumDims(),
                                 op.lowerBoundsMap().getNumSymbols(),
                                 newLowerBounds, op.getContext());
  auto newUpper = AffineMap::get(op.upperBoundsMap().getNumDims(),
                                 op.upperBoundsMap().getNumSymbols(),
                                 newUpperBounds, op.getContext());
  op.lowerBoundsMapAttr(AffineMapAttr::get(newLower));
  op.lowerBoundsGroupsAttr(builder.getI32TensorAttr(groups));
  op.upperBoundsMapAttr(AffineMapAttr::get(newUpper));
  op.upperBoundsGroupsAttr(builder.getI32TensorAttr(groups));
  op.setSteps(newSteps);
}

void denestLoops(mlir::AffineParallelOp op) {
  auto *body = op.getBody();
  auto inner = dyn_cast<AffineParallelOp>(body->front());
  if (!inner) {
    // First op isn't another AffineParallel
    return;
  }

  // Don't de-nest any affine.parallel loop with tags.
  if (hasTags(op) || hasTags(inner))
    return;

  auto yield = cast<AffineYieldOp>(body->back());
  if (yield.operands() != inner.results()) {
    // Fail if inner results is not equal to yield operands.
    // It would be more robust to handle the case where the order was permuted,
    // but it's more complicated and not commonly useful.
    return;
  }
  if (inner.reductions() != op.reductions()) {
    // Verify reductions match (sum of max is cannont be denested)
    return;
  }
  // Because we have already normalized things, we can presume upper bounds are
  // simple constant values.  Gather them
  auto outerRanges = op.upperBoundsMap().getConstantResults();
  auto innerRanges = inner.upperBoundsMap().getConstantResults();
  // Merge them together
  SmallVector<int64_t, 6> newRanges;
  newRanges.insert(newRanges.end(), outerRanges.begin(), outerRanges.end());
  newRanges.insert(newRanges.end(), innerRanges.begin(), innerRanges.end());
  // Extract reductions
  SmallVector<arith::AtomicRMWKind, 8> reductions;
  for (APInt value : op.reductions().getAsValueRange<IntegerAttr>()) {
    reductions.push_back(*arith::symbolizeAtomicRMWKind(value.getZExtValue()));
  }
  // Make a new AffineParallel right before the current op
  OpBuilder builder(op);
  auto newOp = builder.create<AffineParallelOp>(
      op.getLoc(), op.getResultTypes(), reductions, newRanges);
  // Move the deep interior across
  auto &destOps = newOp.getBody()->getOperations();
  destOps.splice(destOps.begin(), inner.getBody()->getOperations());
  // Hook up the block arguments
  for (auto arg : op.getIVs()) {
    arg.replaceAllUsesWith(newOp.getIVs()[arg.getArgNumber()]);
  }
  size_t offset = op.getIVs().size();
  for (auto arg : inner.getIVs()) {
    arg.replaceAllUsesWith(newOp.getIVs()[offset + arg.getArgNumber()]);
  }
  // Erase the old ops
  op.replaceAllUsesWith(newOp);
  op.erase();
}

struct AffineNormalizePass : public AffineNormalizeBase<AffineNormalizePass> {
  AffineNormalizePass() = default;
  explicit AffineNormalizePass(bool promote, bool denest) {
    this->promote = promote;
    this->denest = denest;
  }
  void runOnOperation() override {
    getOperation().walk(normalizeAffineParallel);
    getOperation().walk(elideSingleIterationIndexes);
    if (promote.getValue()) {
      getOperation().walk(promoteIfEmptyIVs);
    }
    if (denest.getValue()) {
      getOperation().walk(denestLoops);
    }
  }
};

std::unique_ptr<Pass> createAffineNormalizePass() {
  return std::make_unique<AffineNormalizePass>();
}

std::unique_ptr<Pass> createAffineNormalizePass(bool promote, bool denest) {
  return std::make_unique<AffineNormalizePass>(promote, denest);
}

} // namespace pmlc::dialect::pxa
