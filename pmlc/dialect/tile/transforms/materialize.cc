// Copyright 2020, Intel Corporation

#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/pass_detail.h"
#include "pmlc/util/interfaces.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::tile {

namespace {

struct MaterializePass : public MaterializeBase<MaterializePass> {
  void runOnFunction() final {
    auto func = getFunction();
    func.walk([&](util::MaterializeOperandsOpInterface op) {
      OpBuilder builder(op);
      if (failed(op.materializeOperands(builder))) {
        op.emitOpError("Failed to materialize operands");
        signalPassFailure();
        return;
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> createMaterializePass() {
  return std::make_unique<MaterializePass>();
}

} // namespace pmlc::dialect::tile
