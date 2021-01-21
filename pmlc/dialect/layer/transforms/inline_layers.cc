// Copyright 2020, Intel Corporation

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/InliningUtils.h"

#include "pmlc/dialect/layer/ir/ops.h"
#include "pmlc/dialect/layer/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::layer {

namespace {

struct InlinerImpl : InlinerInterface {
  using InlinerInterface::InlinerInterface;

  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToReplace) const final {
    IVLOG(1, "handleTerminator");
    auto returnOp = cast<ReturnOp>(op);
    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToReplace.size());
    for (auto item : llvm::zip(valuesToReplace, returnOp.getOperands())) {
      Value oldValue, newValue;
      std::tie(oldValue, newValue) = item;
      oldValue.replaceAllUsesWith(newValue);
    }
  }

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    return true;
  }
};

struct InlineLayersPass : public InlineLayersBase<InlineLayersPass> {
  void runOnFunction() final {
    auto func = getFunction();
    InlinerImpl inliner(&getContext());
    func.walk([&](BoxOp op) {
      if (failed(inlineRegion(/*interface=*/inliner,
                              /*src=*/&op.body(),
                              /*inlinePoint=*/op,
                              /*inlinedOperands=*/op.operands(),
                              /*resultsToReplace=*/op.results(),
                              /*inlineLoc=*/op.getLoc()))) {
        op.emitOpError("failed to be inlined");
        signalPassFailure();
        return;
      }
      op.erase();
    });
  }
};

} // namespace

std::unique_ptr<Pass> createInlineLayersPass() {
  return std::make_unique<InlineLayersPass>();
}

} // namespace pmlc::dialect::layer
