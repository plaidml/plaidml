// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/transforms/constant_types.h"

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

PatternMatchResult ConstantTypesRewriter::matchAndRewrite(ScalarConstantOp constOp, PatternRewriter& rewriter) const {
  IVLOG(1, "ConstantTypesPass::matchAndRewrite");
  IVLOG(1, "float_x " << static_cast<int>(floatx_));
  IVLOG(1, "int " << static_cast<int>(intx_));

  return matchSuccess();
}

/*
void ConstantTypesPass::runOnFunction() {
  IVLOG(1, "ConstantTypesPass::runOnFunction");
  IVLOG(1, "float_x " << static_cast<int>(floatx_));
  IVLOG(1, "int " << static_cast<int>(intx_));

  // TODO: check compute_shape pass

  if (floatx_ != DataType::invalid && !isFloat(floatx_)) {
    throw std::runtime_error("floatx is not floating-point type ");
  }
  if (intx_ != DataType::invalid && !isInteger(intx_)) {
    throw std::runtime_error("intx is not integer");
  }

  auto func = getFunction();
  func.walk([this](pmlc::dialect::eltwise::ScalarConstantOp* op) {
    try {
      mlir::Builder builder(&getContext());

      if (auto int_attr = op->getIntAttr()) {
        int64_t value = int_attr.getInt();
        IVLOG(1, "value " << value);

        switch (intx_) {
          case DataType::u1:
          default:
            break;
        }
      } else if (auto float_attr = op->getFloatAttr()) {
        double value = float_attr.getValueAsDouble();
        IVLOG(1, "value " << value);

      }
    } catch (const std::exception& ex) {
      op->emitError(ex.what());
      signalPassFailure();
    }
  });
}  // namespace pmlc::dialect::tile
*/

std::unique_ptr<mlir::Pass> createConstantTypesPass(const DataType& floatx, const DataType& intx) {
  auto pass = std::make_unique<ConstantTypesPass>(floatx, intx);

  IVLOG(1, "Creating pass with floatx " << static_cast<int>(floatx));
  IVLOG(1, "Creating pass with intx " << static_cast<int>(intx));

  IVLOG(1, "Created pass with floatx " << static_cast<int>(pass->floatx_));
  IVLOG(1, "Created pass with intx " << static_cast<int>(pass->intx_));

  return pass;
}

}  // namespace pmlc::dialect::tile
