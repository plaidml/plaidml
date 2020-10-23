// Copyright 2020, Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"

#include "pmlc/dialect/abi/ir/dialect.h"

namespace pmlc::dialect::abi {

namespace {

struct DenormalizeConstantsPattern final
    : public mlir::OpRewritePattern<LoopOp> {
  using mlir::OpRewritePattern<LoopOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(LoopOp loopOp, mlir::PatternRewriter &rewriter) const final {
    mlir::OpBuilder::InsertionGuard insertionGuard{rewriter};
    auto networkOp = loopOp.getInitTerminator();
    auto networkFieldTypes = loopOp.getAttrOfType<mlir::ArrayAttr>(
        abi::LoopOp::getNetworkFieldTypesAttrName());
    mlir::SmallVector<mlir::Attribute, 8> newNetworkFieldTypes;
    rewriter.startRootUpdate(loopOp);
    rewriter.setInsertionPointToStart(&loopOp.bodyRegion().front());
    unsigned idx = 0;
    for (auto tyAttr : networkFieldTypes) {
      auto val = networkOp.getOperand(idx);
      if (auto constOp = val.getDefiningOp<mlir::ConstantOp>()) {
        auto denormOp = constOp.clone();
        rewriter.insert(denormOp);
        auto arg = loopOp.bodyEntryBlock()->getArgument(idx);
        arg.replaceAllUsesWith(denormOp);
        loopOp.bodyEntryBlock()->eraseArgument(idx);
        networkOp.getOperation()->eraseOperand(idx);
      } else {
        newNetworkFieldTypes.emplace_back(tyAttr);
        ++idx;
      }
    }
    if (networkFieldTypes.size() != newNetworkFieldTypes.size()) {
      loopOp.setAttr(abi::LoopOp::getNetworkFieldTypesAttrName(),
                     rewriter.getArrayAttr(newNetworkFieldTypes));
      rewriter.finalizeRootUpdate(loopOp);
      return mlir::success();
    }
    rewriter.cancelRootUpdate(loopOp);
    return mlir::failure();
  }
};

} // namespace

void LoopOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *ctx) {
  patterns.insert<DenormalizeConstantsPattern>(ctx);
}

mlir::Region &LoopOp::getLoopBody() { return bodyRegion(); }

bool LoopOp::isDefinedOutsideOfLoop(mlir::Value value) {
  auto blockArg = value.dyn_cast<mlir::BlockArgument>();
  if (!blockArg) {
    // Regular values are always defined inside the loop.
    return false;
  }
  if (blockArg.getParentBlock() != &bodyRegion().front()) {
    // The only block arguments that come from outside the loop come from the
    // entry region.
    return false;
  }
  // Finally: if the argument number is less than the number of items in the
  // network type, the value is coming from outside of the loop region via the
  // network pointer (making its user eligible for hoisting); otherwise, the
  // argument is coming from what will eventually be a parameter to the loop
  // iterator call, so it must not be hoisted.
  return blockArg.getArgNumber() < getNumNetworkFields();
}

LogicalResult LoopOp::moveOutOfLoop(mlir::ArrayRef<mlir::Operation *> ops) {
  auto networkOp = getInitTerminator();
  mlir::Block *bodyEntryBlock = &bodyRegion().front();
  for (auto op : ops) {
    for (auto &operand : op->getOpOperands()) {
      auto blockArg = operand.get().dyn_cast<mlir::BlockArgument>();
      if (!blockArg || blockArg.getParentBlock() != bodyEntryBlock ||
          networkOp.getNumOperands() <= blockArg.getArgNumber()) {
        // N.B. It's unclear that this should ever happen (where could the
        // operand have come from, if we're hoisting it?), but if it does,
        // we handle it.
        continue;
      }
      operand.set(networkOp.getOperand(blockArg.getArgNumber()));
      if (blockArg.use_empty()) {
        // After hoisting this op to the init region, we will no longer need
        // this body argument.
        networkOp.getOperation()->eraseOperand(blockArg.getArgNumber());
        bodyEntryBlock->eraseArgument(blockArg.getArgNumber());
      }
    }
    for (auto result : op->getResults()) {
      if (result.use_empty()) {
        continue;
      }
      unsigned idx = networkOp.getNumOperands();
      auto plumbedValue = bodyEntryBlock->insertArgument(idx, result.getType());
      result.replaceAllUsesWith(plumbedValue);
      networkOp.getOperation()->insertOperands(idx, mlir::ValueRange{result});
    }
    op->moveBefore(networkOp);
  }
  setNetworkFieldTypes(networkOp.getOperandTypes());
  return mlir::success();
}

} // namespace pmlc::dialect::abi
