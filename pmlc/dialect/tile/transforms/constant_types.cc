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

static llvm::cl::opt<std::string> constant_types_option_floatx("tile-constant-types-floatx", llvm::cl::init("float32"),
                                                               llvm::cl::desc("set floating-point constant precision"),
                                                               llvm::cl::cat(constant_types_options));

static llvm::cl::opt<std::string> constant_types_option_intx("tile-constant-types-intx", llvm::cl::init("int32"),
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
  IVLOG(3, "ConstantTypesPass::matchAndRewrite");

  auto type = constOp.getType();
  auto shape = eltwise::getRankedTensorType(type).getShape();

  auto float_attr = constOp.getFloatAttr();
  auto int_attr = constOp.getIntAttr();

  if (float_attr && floatx_ != DataType::invalid) {
    double value = float_attr.getValueAsDouble();
    auto elementType = ScalarType::get(type.getContext(), floatx_);
    auto new_type = RankedTensorType::get(shape, elementType);

    if (new_type == type) {
      return matchFailure();
    }
    rewriter.replaceOpWithNewOp<ScalarConstantOp>(constOp, elementType, value);
    return matchSuccess();

  } else if (int_attr && intx_ != DataType::invalid) {
    int64_t value = int_attr.getInt();

    if (pmlc::util::isUnsigned(intx_) && (value < 0)) {
      std::stringstream ss;
      ss << "Invalid datatype for negative constant";
      throw std::runtime_error(ss.str());
    }
    auto elementType = ScalarType::get(type.getContext(), intx_);
    auto new_type = RankedTensorType::get(shape, elementType);

    if (new_type == type) {
      return matchFailure();
    }
    rewriter.replaceOpWithNewOp<ScalarConstantOp>(constOp, elementType, value);
    return matchSuccess();
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
  };

  void runOnOperation() final;

  DataType floatx_;
  DataType intx_;
};

void ConstantTypesPass::runOnOperation() {
  OwningRewritePatternList patterns;
  patterns.insert<ConstantTypesRewriter>(&getContext(), floatx_, intx_);
  applyPatternsGreedily(getOperation()->getRegions(), patterns);
}

std::unique_ptr<mlir::Pass> createConstantTypesPass(const DataType& floatx, const DataType& intx) {
  const auto& floatx_str = pmlc::util::stringifyDataType(floatx);
  const auto& intx_str = pmlc::util::stringifyDataType(intx);
  return std::make_unique<ConstantTypesPass>(floatx_str, intx_str);
}

static mlir::PassRegistration<ConstantTypesPass> constant_types_pass("tile-constant-types",
                                                                     "Set constant types precision");

}  // namespace pmlc::dialect::tile
