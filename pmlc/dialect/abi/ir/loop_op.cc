// Copyright 2020, Intel Corporation

#include "pmlc/dialect/abi/ir/dialect.h"

namespace pmlc::dialect::abi {

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
