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

static llvm::cl::opt<std::string> constant_types_option_floatx("tile-constant-types-floatx", llvm::cl::init("f32"),
                                                               llvm::cl::desc("set floating-point constant precision"),
                                                               llvm::cl::cat(constant_types_options));

static llvm::cl::opt<std::string> constant_types_option_intx("tile-constant-types-intx", llvm::cl::init("i32"),
                                                             llvm::cl::desc("set integer constant precision"),
                                                             llvm::cl::cat(constant_types_options));

static DataType parse_command_line_opt(const llvm::cl::opt<std::string>& option) {
  auto opt = pmlc::util::symbolizeDataType(option);
  if (!opt.hasValue()) {
    std::stringstream ss;
    ss << " Invalid runtime option " << option;
    throw std::runtime_error(ss.str());
  }
  return opt.getValue();
}

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
  }
  if (int_attr && intx_ != DataType::invalid) {
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
  }
  return matchFailure();
}

struct ConstantTypesPass : public OperationPass<ConstantTypesPass> {
  ConstantTypesPass(DataType floatx = parse_command_line_opt(constant_types_option_floatx),
                    DataType intx = parse_command_line_opt(constant_types_option_intx))
      : floatx_(floatx), intx_(intx) {}

  void runOnOperation() final;

  DataType floatx_;
  DataType intx_;
};

void ConstantTypesPass::runOnOperation() {
  OwningRewritePatternList patterns;
  patterns.insert<ConstantTypesRewriter>(&getContext(), floatx_, intx_);
  applyPatternsGreedily(getOperation()->getRegions(), patterns);
}

std::unique_ptr<mlir::Pass> createConstantTypesPass(DataType floatx, DataType intx) {
  return std::make_unique<ConstantTypesPass>(floatx, intx);
}

static mlir::PassRegistration<ConstantTypesPass> constant_types_pass("tile-constant-types",
                                                                     "Set constant types precision");

}  // namespace pmlc::dialect::tile
