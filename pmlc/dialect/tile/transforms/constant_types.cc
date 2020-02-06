// Copyright 2020 Intel Corporation

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

static llvm::cl::OptionCategory optionCategory("tile-constant-types options");

static llvm::cl::opt<std::string>
    clFloatxOption("tile-constant-types-floatx", llvm::cl::init("f32"),
                   llvm::cl::desc("set floating-point constant precision"),
                   llvm::cl::cat(optionCategory));

static llvm::cl::opt<std::string>
    clIntxOption("tile-constant-types-intx", llvm::cl::init("i32"),
                 llvm::cl::desc("set integer constant precision"),
                 llvm::cl::cat(optionCategory));

static DataType parseOption(const llvm::cl::opt<std::string> &option) {
  auto opt = pmlc::util::symbolizeDataType(option);
  if (!opt.hasValue()) {
    throw std::runtime_error(
        llvm::formatv("Invalid runtime option: '{0}'", option));
  }
  return opt.getValue();
}

struct ConstantTypesRewriter : public OpRewritePattern<ScalarConstantOp> {
  ConstantTypesRewriter(mlir::MLIRContext *context, DataType floatx,
                        DataType intx)
      : OpRewritePattern<ScalarConstantOp>(context), floatx(floatx),
        intx(intx) {}

  PatternMatchResult matchAndRewrite(ScalarConstantOp constOp,
                                     PatternRewriter &rewriter) const override {
    IVLOG(3, "ConstantTypesPass::matchAndRewrite");

    auto type = constOp.getType();
    auto shape = eltwise::getRankedTensorType(type).getShape();

    auto floatAttr = constOp.getFloatAttr();
    if (floatAttr && floatx != DataType::invalid) {
      double value = floatAttr.getValueAsDouble();
      auto elementType = ScalarType::get(type.getContext(), floatx);
      auto tensorType = RankedTensorType::get(shape, elementType);
      if (tensorType == type) {
        return matchFailure();
      }
      rewriter.replaceOpWithNewOp<ScalarConstantOp>(constOp, elementType,
                                                    value);
      return matchSuccess();
    }

    auto intAttr = constOp.getIntAttr();
    if (intAttr && intx != DataType::invalid) {
      int64_t value = intAttr.getInt();
      if (pmlc::util::isUnsigned(intx) && (value < 0)) {
        throw std::runtime_error("Invalid datatype for negative constant");
      }
      auto elementType = ScalarType::get(type.getContext(), intx);
      auto tensorType = RankedTensorType::get(shape, elementType);
      if (tensorType == type) {
        return matchFailure();
      }
      rewriter.replaceOpWithNewOp<ScalarConstantOp>(constOp, elementType,
                                                    value);
      return matchSuccess();
    }

    return matchFailure();
  }

  DataType floatx;
  DataType intx;
};

struct ConstantTypesPass : public OperationPass<ConstantTypesPass> {
  ConstantTypesPass() = default;
  ConstantTypesPass(DataType floatx, DataType intx)
      : floatx(floatx), intx(intx) {}

  void runOnOperation() final {
    OwningRewritePatternList patterns;
    patterns.insert<ConstantTypesRewriter>(&getContext(), floatx, intx);
    applyPatternsGreedily(getOperation()->getRegions(), patterns);
  }

  DataType floatx = parseOption(clFloatxOption);
  DataType intx = parseOption(clIntxOption);
};

std::unique_ptr<mlir::Pass> createConstantTypesPass(DataType floatx,
                                                    DataType intx) {
  return std::make_unique<ConstantTypesPass>(floatx, intx);
}

static mlir::PassRegistration<ConstantTypesPass>
    pass("tile-constant-types", "Set constant types precision");

} // namespace pmlc::dialect::tile
