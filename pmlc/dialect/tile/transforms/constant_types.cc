// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/transforms/constant_types.h"

#include <limits>
#include <list>
#include <memory>

#include "llvm/Support/FormatVariadic.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
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

      mlir::Builder builder(&getContext());

      auto value = op.getValue();
      value.dump();

      if (auto int_attr = op.getIntAttr()) {
        IVLOG(1, "got int_attr");
        int_attr.dump();

        int64_t value = int_attr.getInt();
        if (intx_.getWidth() == 32) {
          op.setValue(builder.getI32IntegerAttr(value));
        } else if (intx_.getWidth() == 64) {
          op.setValue(builder.getI64IntegerAttr(value));
        } else {
          IVLOG(1, "Warning: unknown integer width " << floatx_.getWidth());
        }

      } else if (auto float_attr = op.getFloatAttr()) {
        IVLOG(1, "got float_attr");

        double value = float_attr.getValueAsDouble();
        IVLOG(1, "got value " << value);

        if (floatx_.getWidth() == 32) {
          IVLOG(1, "floatx_.getWidth() == 32");
          op.setValue(builder.getF32FloatAttr(value));
        } else if (floatx_.getWidth() == 64) {
          IVLOG(1, "floatx_.getWidth() == 64");
          op.setValue(builder.getF64FloatAttr(value));
        } else {
          IVLOG(1, "Warning: unknown float width " << floatx_.getWidth());
        }
      }

    } catch (const std::exception& ex) {
      op.emitError(ex.what());
      signalPassFailure();
    }
  });
}

std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> createConstantTypesPass(mlir::FloatType floatx,
                                                                        mlir::IntegerType intx) {
  auto const_pass = std::make_unique<ConstantTypesPass>();

  IVLOG(1, "Creating createConstantTypesPass");
  IVLOG(1, "with float width " << floatx.getWidth());
  const_pass->floatx_ = floatx;
  const_pass->intx_ = intx;

  std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> pass = std::move(const_pass);
  return pass;
}

}  // namespace pmlc::dialect::tile
