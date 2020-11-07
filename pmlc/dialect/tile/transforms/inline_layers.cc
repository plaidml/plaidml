// Copyright 2020, Intel Corporation

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/InliningUtils.h"

#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::tile {

namespace {

struct InlinerImpl : InlinerInterface {
  using InlinerInterface::InlinerInterface;

  bool isLegalToInline(Operation *op, Region *region,
                       BlockAndValueMapping &valueMapping) const final {
    return true;
  }

  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToReplace) const final {
    auto returnOp = cast<LayerReturnOp>(op);
    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToReplace.size());
    for (auto item : llvm::zip(valuesToReplace, returnOp.getOperands())) {
      Value oldValue, newValue;
      std::tie(oldValue, newValue) = item;
      oldValue.replaceAllUsesWith(newValue);
    }
  }
};

struct InlineLayersPass : public InlineLayersBase<InlineLayersPass> {
  void runOnFunction() final {
    auto func = getFunction();
    InlinerImpl inliner(&getContext());
    func.walk([&](LayerOp op) {
      if (failed(inlineRegion(inliner, &op.body(), op, op.operands(),
                              op.results(), op.getLoc(),
                              /*shouldCloneInlinedRegion=*/true))) {
        op.emitOpError("Failed to inline layer op");
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

} // namespace pmlc::dialect::tile
