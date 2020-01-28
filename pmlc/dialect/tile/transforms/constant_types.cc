// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/transforms/constant_types.h"

#include <memory>

#include "llvm/Support/FormatVariadic.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/eltwise/ir/util.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/util/enums.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

namespace pmlc::dialect::tile {

PatternMatchResult ConstantTypesRewriter::matchAndRewrite(ScalarConstantOp constOp, PatternRewriter& rewriter) const {
  IVLOG(1, "ConstantTypesPass::matchAndRewrite");
  IVLOG(1, "float_x " << static_cast<int>(floatx_));
  IVLOG(1, "int " << static_cast<int>(intx_));

  auto type = constOp.getType();
  IVLOG(2, "ConstantTypesRewriter::type> " << mlir::debugString(type));

  auto shape = pmlc::dialect::eltwise::getRankedTensorType(type).getShape();

  // auto cur_type = pmlc::dialect::eltwise::getRankedTensorType(type);
  // IVLOG(2, "ConstantTypesRewriter::cur_type> " << mlir::debugString(cur_type));
  // auto scalar_type = type.cast<eltwise::ScalarType>();
  // IVLOG(2, "ConstantTypesRewriter::scalar_type> " << mlir::debugString(scalar_type));

  if (auto float_attr = constOp.getFloatAttr()) {
    double value = float_attr.getValueAsDouble();
    auto elementType = ScalarType::get(type.getContext(), floatx_);
    auto new_type = RankedTensorType::get(shape, elementType);
    IVLOG(2, "ConstantTypesRewriter::new_type> " << mlir::debugString(new_type));

    if (new_type == type) {
      IVLOG(1, "return match failure");
      return matchFailure();
    } else {
      rewriter.replaceOpWithNewOp<ScalarConstantOp>(constOp, elementType, value);
      IVLOG(1, "return match success");
      return matchSuccess();
    }

  } else if (auto int_attr = constOp.getIntAttr()) {
    int64_t value = int_attr.getInt();

    if (pmlc::util::isUnsigned(intx_) && (value < 0)) {
      std::stringstream ss;
      ss << "Invalid datatype for negative constant";
      throw std::runtime_error(ss.str());
    }
    auto elementType = ScalarType::get(type.getContext(), intx_);
    auto new_type = RankedTensorType::get(shape, elementType);
    IVLOG(2, "ConstantTypesRewriter::new_type> " << mlir::debugString(new_type));

    if (new_type == type) {
      IVLOG(1, "return match failure");
      return matchFailure();
    } else {
      rewriter.replaceOpWithNewOp<ScalarConstantOp>(constOp, elementType, value);
      IVLOG(1, "return match success");
      return matchSuccess();
    }
  } else {
    IVLOG(1, "Warning: non-float, non-int type");
  }
  return matchFailure();
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
