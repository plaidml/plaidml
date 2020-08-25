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

using namespace mlir; // NOLINT

// TODO: fix upstream
void normalizeAffineParallel(AffineParallelOp op);

namespace pmlc::dialect::pxa {

/// Promotes the loop body of an affine.parallel to its containing block if no
/// induction variables are present.
void promoteIfEmptyIVs(AffineParallelOp op) {
  // Nothing to do when there are induction variables.
  if (op.getNumDims())
    return;

  // Only remove ops that don't have any custom attributes (i.e. those not
  // defined by the op itself). This is needed to ensure that we don't drop
  // ops that are structural in nature. One case is for kernel outlining; it's
  // possible to have an outermost loop with a single iteration which would
  // represent a single kernel launch. In this case, we don't want this
  // canonicalization to drop it.
  // TODO: This feels like a hack; should we create a new op with a region for
  // this case? Or perhaps we can have a standard attribute that is used for
  // controlling canoncializations?
  StringSet<> opAttrs{AffineParallelOp::getReductionsAttrName(),
                      AffineParallelOp::getLowerBoundsMapAttrName(),
                      AffineParallelOp::getUpperBoundsMapAttrName(),
                      AffineParallelOp::getStepsAttrName()};
  for (NamedAttribute attr : op.getAttrs())
    if (!opAttrs.count(attr.first.strref()))
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
  AffineValueMap ranges = op.getRangesValueMap();
  Block *body = op.getBody();
  SmallVector<AffineExpr, 6> newLowerBounds;
  SmallVector<AffineExpr, 6> newUpperBounds;
  SmallVector<int64_t, 6> newSteps;
  SmallVector<BlockArgument, 6> argsToRemove;
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
  op.upperBoundsMapAttr(AffineMapAttr::get(newUpper));
  op.setSteps(newSteps);
}

struct AffineNormalizePass : public AffineNormalizeBase<AffineNormalizePass> {
  void runOnFunction() override {
    getFunction().walk(::normalizeAffineParallel);
    getFunction().walk(elideSingleIterationIndexes);
    getFunction().walk(promoteIfEmptyIVs);
  }
};

std::unique_ptr<Pass> createAffineNormalizePass() {
  return std::make_unique<AffineNormalizePass>();
}

} // namespace pmlc::dialect::pxa
