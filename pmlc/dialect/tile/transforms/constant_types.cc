// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/transforms/constant_types.h"

#include <limits>
#include <list>
#include <memory>

#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

namespace pmlc::dialect::tile {

void ConstantTypesPass::runOnFunction() {
  auto func = getFunction();
  IVLOG(1, "ConstantTypesPass::runOnFunction");
  func.walk([this](pmlc::dialect::eltwise::ScalarConstantOp op) {
    try {
      IVLOG(1, "ConstantTypesPass found ScalarConstantOp");

      auto value = op.getValue();
      value.dump();

      if (auto int_attr = op.getIntAttr()) {
        IVLOG(1, "got int_attr");
        int_attr.dump();
      } else if (auto float_attr = op.getFloatAttr()) {
        IVLOG(1, "got float_attr");
        double value = float_attr.getValueAsDouble();

        float_attr.dump();
      }

      // ConstantTypes impl(op);
      /* auto maps = llvm::makeArrayRef(impl.affineMaps);
       op.setLowerBounds(impl.lowerBounds);
       op.setUpperBounds(impl.upperBounds);
       op.setSink(maps.front());
       op.setSources(maps.drop_front());
       op.setConstraints(impl.getConstraints()); */
    } catch (const std::exception& ex) {
      op.emitError(ex.what());
      signalPassFailure();
    }
  });
}

std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> createConstantTypesPass(mlir::FloatType floatx,
                                                                        mlir::IntegerType intx) {
  auto const_pass = std::make_unique<ConstantTypesPass>();
  const_pass->floatx_ = floatx;
  const_pass->intx_ = intx;

  std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> pass = std::move(const_pass);
  return pass;
}

}  // namespace pmlc::dialect::tile
