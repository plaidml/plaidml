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
    auto networkFieldTypes = loopOp.networkFieldTypes();
    mlir::SmallVector<mlir::Attribute, 8> newNetworkFieldTypes;
    rewriter.startRootUpdate(loopOp);
    rewriter.setInsertionPointToStart(&loopOp.bodyRegion().front());
    unsigned idx = 0;
    for (auto tyAttr : networkFieldTypes) {
      auto val = networkOp.getOperand(idx);
      auto definingOp = val.getDefiningOp();
      if (definingOp && definingOp->hasTrait<mlir::OpTrait::ConstantLike>()) {
        // N.B. Constant-Like operations have no side-effects, exactly one
        //      result, and no operands.
        auto *denormOp = definingOp->clone();
        rewriter.insert(denormOp);
        auto arg = loopOp.getBodyEntryBlock()->getArgument(idx);
        arg.replaceAllUsesWith(denormOp->getResult(0));
        loopOp.getBodyEntryBlock()->eraseArgument(idx);
        loopOp.getFiniEntryBlock()->eraseArgument(idx);
        networkOp.getOperation()->eraseOperand(idx);
      } else {
        newNetworkFieldTypes.emplace_back(tyAttr);
        ++idx;
      }
    }
    if (networkFieldTypes.size() != newNetworkFieldTypes.size()) {
      loopOp.networkFieldTypesAttr(rewriter.getArrayAttr(newNetworkFieldTypes));
      rewriter.finalizeRootUpdate(loopOp);
      return mlir::success();
    }
    rewriter.cancelRootUpdate(loopOp);
    return mlir::failure();
  }
};

struct RemoveUnusedNetworkFieldsPattern final
    : public mlir::OpRewritePattern<LoopOp> {
  using mlir::OpRewritePattern<LoopOp>::OpRewritePattern;

  mlir::LogicalResult match(LoopOp loopOp) const final {
    auto networkOp = loopOp.getInitTerminator();
    auto *bodyEntryBlock = loopOp.getBodyEntryBlock();
    auto *finiEntryBlock = loopOp.getFiniEntryBlock();
    for (unsigned idx = 0; idx < networkOp.getNumOperands(); ++idx) {
      if (bodyEntryBlock->getArgument(idx).use_empty() &&
          finiEntryBlock->getArgument(idx).use_empty()) {
        return mlir::success();
      }
    }
    return mlir::failure();
  }

  void rewrite(LoopOp loopOp, mlir::PatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(loopOp, [&] {
      auto networkOp = loopOp.getInitTerminator();
      auto *bodyEntryBlock = loopOp.getBodyEntryBlock();
      auto *finiEntryBlock = loopOp.getFiniEntryBlock();
      unsigned idx = 0;
      while (idx < networkOp.getNumOperands()) {
        if (bodyEntryBlock->getArgument(idx).use_empty() &&
            finiEntryBlock->getArgument(idx).use_empty()) {
          networkOp.getOperation()->eraseOperand(idx);
          bodyEntryBlock->eraseArgument(idx);
          finiEntryBlock->eraseArgument(idx);
        } else {
          ++idx;
        }
      }
      loopOp.setNetworkFieldTypes(networkOp.getOperandTypes());
    });
  }
};

} // namespace

void LoopOp::build(::mlir::OpBuilder &odsBuilder,
                   ::mlir::OperationState &odsState) {
  odsState.addAttribute("networkFieldTypes", odsBuilder.getTypeArrayAttr({}));
  odsState.addRegion();
  odsState.addRegion();
  odsState.addRegion();
}

std::vector<mlir::Type> LoopOp::getNetworkFieldTypes() {
  std::vector<mlir::Type> result;
  auto arrayAttr = networkFieldTypes();
  if (arrayAttr) {
    for (auto attr : arrayAttr) {
      if (auto tyAttr = attr.dyn_cast<mlir::TypeAttr>()) {
        result.emplace_back(tyAttr.getValue());
      }
    }
  }
  return result;
}

unsigned LoopOp::getNumNetworkFields() {
  auto arrayAttr = networkFieldTypes();
  if (arrayAttr) {
    return arrayAttr.size();
  }
  return 0;
}

void LoopOp::setNetworkFieldTypes(mlir::TypeRange types) {
  mlir::SmallVector<mlir::Attribute, 8> attrs;
  for (auto ty : types) {
    attrs.emplace_back(mlir::TypeAttr::get(ty));
  }
  auto arrayAttr = mlir::ArrayAttr::get(attrs, getContext());
  networkFieldTypesAttr(arrayAttr);
}

mlir::Block *LoopOp::getBodyEntryBlock() { return &bodyRegion().front(); }
mlir::Block *LoopOp::getFiniEntryBlock() { return &finiRegion().front(); }

YieldOp LoopOp::getInitTerminator() {
  return mlir::cast<YieldOp>(initRegion().back().getTerminator());
}

TerminatorOp LoopOp::getFiniTerminator() {
  return mlir::cast<TerminatorOp>(finiRegion().back().getTerminator());
}

void LoopOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *ctx) {
  patterns.insert<DenormalizeConstantsPattern>(ctx);
  patterns.insert<RemoveUnusedNetworkFieldsPattern>(ctx);
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
  mlir::Block *bodyEntryBlock = getBodyEntryBlock();
  mlir::Block *finiEntryBlock = getFiniEntryBlock();
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
    }
    for (auto result : op->getResults()) {
      if (result.use_empty()) {
        continue;
      }
      unsigned idx = networkOp.getNumOperands();
      finiEntryBlock->insertArgument(idx, result.getType());
      auto plumbedValue = bodyEntryBlock->insertArgument(idx, result.getType());
      result.replaceAllUsesWith(plumbedValue);
      networkOp.getOperation()->insertOperands(idx, mlir::ValueRange{result});
    }
    op->moveBefore(networkOp);
  }
  setNetworkFieldTypes(networkOp.getOperandTypes());
  return mlir::success();
}

LogicalResult LoopOp::moveToLoopEpilog(mlir::ArrayRef<mlir::Operation *> ops) {
  auto networkOp = getInitTerminator();
  auto finiTerminator = getFiniTerminator();
  mlir::Block *bodyEntryBlock = getBodyEntryBlock();
  mlir::Block *finiEntryBlock = getFiniEntryBlock();
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
      operand.set(finiEntryBlock->getArgument(blockArg.getArgNumber()));
    }
    for (auto result : op->getResults()) {
      if (result.use_empty()) {
        continue;
      }
      // If the results of the operation are in use, we don't know how to
      // hoist it.
      return mlir::failure();
    }
    op->moveBefore(finiTerminator);
  }
  return mlir::success();
}

} // namespace pmlc::dialect::abi
