// Copyright 2020 Intel Corporation

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"

#include "pmlc/dialect/tile/transforms/pass_detail.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::tile {

namespace {

class MakeProgramDriver : public PatternRewriter {
public:
  MakeProgramDriver(MLIRContext *ctx, const OwningRewritePatternList &patterns)
      : PatternRewriter(ctx), matcher(patterns), folder(ctx) {
    matcher.applyDefaultCostModel();
  }

  void run(FuncOp funcOp) {
    funcOp.walk([&](Operation *op) {
      // Try to fold this op.
      if (succeeded(folder.tryToFold(op))) {
        return;
      }

      // Make sure that any new operations are inserted at this point.
      setInsertionPoint(op);

      // Try to match one of the patterns.
      matcher.matchAndRewrite(op, *this);
    });
  }

private:
  PatternApplicator matcher;
  OperationFolder folder;
};

struct MakeProgramPass
    : public mlir::PassWrapper<MakeProgramPass, mlir::FunctionPass> {
  void runOnFunction() final {
    OwningRewritePatternList patterns;
    auto context = &getContext();
    for (auto op : context->getRegisteredOperations()) {
      op->getCanonicalizationPatterns(patterns, context);
    }

    MakeProgramDriver driver(context, patterns);
    driver.run(getFunction());
  }

  static std::unique_ptr<mlir::Pass> create() {
    return std::make_unique<MakeProgramPass>();
  }
};

} // namespace

std::unique_ptr<Pass> createMakeProgramPass() {
  return std::make_unique<MakeProgramPass>();
}

} // namespace pmlc::dialect::tile
