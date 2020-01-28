// Copyright 2019, Intel Corporation

#include <memory>

#include "llvm/Support/FormatVariadic.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/eltwise/ir/util.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/util/enums.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

namespace pmlc::dialect::tile {
using mlir::OperationPass;
using mlir::OpRewritePattern;
using mlir::OwningRewritePatternList;
using mlir::PatternMatchResult;
using mlir::PatternRewriter;

using pmlc::dialect::eltwise::ScalarConstantOp;

static llvm::cl::OptionCategory constant_types_options("tile-constant-types options");

static llvm::cl::opt<std::string> constant_types_option_floatx("tile-constant-types-floatx", llvm::cl::init("float64"),
                                                               llvm::cl::desc("set floating-point constant precision"),
                                                               llvm::cl::cat(constant_types_options));

static llvm::cl::opt<std::string> constant_types_option_intx("tile-constant-types-intx", llvm::cl::init("int64"),
                                                             llvm::cl::desc("set integer constant precision"),
                                                             llvm::cl::cat(constant_types_options));

struct ConstantTypesRewriter : public OpRewritePattern<ScalarConstantOp> {
  ConstantTypesRewriter(mlir::MLIRContext* context, DataType floatx, DataType intx)
      : OpRewritePattern<ScalarConstantOp>(context), floatx_(floatx), intx_(intx) {}

  DataType floatx_;
  DataType intx_;

  PatternMatchResult matchAndRewrite(ScalarConstantOp constOp, PatternRewriter& rewriter) const override;
};

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

struct ConstantTypesPass : public OperationPass<ConstantTypesPass> {
  ConstantTypesPass(const std::string& floatx = constant_types_option_floatx,
                    const std::string& intx = constant_types_option_intx) {
    floatx_ = pmlc::util::from_string(floatx);
    intx_ = pmlc::util::from_string(intx);

    if (floatx_ == DataType::invalid) {
      std::stringstream ss;
      ss << "Invalid floatx option " << floatx;
      throw std::runtime_error(ss.str());
    }
    if (intx_ == DataType::invalid) {
      std::stringstream ss;
      ss << "Invalid intx option " << intx;
      throw std::runtime_error(ss.str());
    }

    IVLOG(1, "creating ConstantTypesPass with floatx " << floatx);
    // IVLOG(1, "floatx_ " << floatx_);
    IVLOG(1, "creating ConstantTypesPass with floatx " << floatx);
    // IVLOG(1, "intx_ " << intx_);

    // TODO set floatx / intx
  };

  void runOnOperation() final;

  DataType floatx_;
  DataType intx_;
};

void ConstantTypesPass::runOnOperation() {
  OwningRewritePatternList patterns;

  patterns.insert<ConstantTypesRewriter>(&getContext(), floatx_, intx_);

  // TODO: Instead of adding all known patterns from the whole system lazily
  // add and cache the canonicalization patterns for ops we see in practice
  // when building the worklist.  For now, we just grab everything.
  // auto* context = &getContext();
  // for (auto* op : context->getRegisteredOperations()) op->getCanonicalizationPatterns//(patterns, context);

  Operation* op = getOperation();
  applyPatternsGreedily(op->getRegions(), patterns);
}

std::unique_ptr<mlir::Pass> createConstantTypesPass(const DataType& floatx, const DataType& intx) {
  std::string floatx_str = pmlc::util::stringifyDataType(floatx);
  std::string intx_str = pmlc::util::stringifyDataType(intx);

  IVLOG(1, "Creating pass with floatx " << static_cast<int>(floatx));
  IVLOG(1, "Creating pass with intx " << static_cast<int>(intx));

  auto pass = std::make_unique<ConstantTypesPass>(floatx_str, intx_str);

  IVLOG(1, "Created pass with floatx_str " << static_cast<int>(pass->floatx_));
  IVLOG(1, "Created pass with intx_str " << static_cast<int>(pass->intx_));

  return pass;
}

static mlir::PassRegistration<ConstantTypesPass> constant_types_pass("tile-constant-types",
                                                                     "Set constant types precision");

}  // namespace pmlc::dialect::tile
