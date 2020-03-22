// Copyright 2020 Intel Corporation

#include <memory>

#include "llvm/Support/FormatVariadic.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

using mlir::failure;
using mlir::IntegerType;
using mlir::LogicalResult;
using mlir::OperationPass;
using mlir::OpRewritePattern;
using mlir::OwningRewritePatternList;
using mlir::PatternRewriter;
using mlir::success;
using mlir::Type;

using eltwise::ScalarConstantOp;

static llvm::cl::OptionCategory optionCategory("tile-constant-types options");

static llvm::cl::opt<std::string>
    clFloatxOption("tile-constant-types-floatx", llvm::cl::init("f32"),
                   llvm::cl::desc("set floating-point constant precision"),
                   llvm::cl::cat(optionCategory));

static llvm::cl::opt<std::string>
    clIntxOption("tile-constant-types-intx", llvm::cl::init("i32"),
                 llvm::cl::desc("set integer constant precision"),
                 llvm::cl::cat(optionCategory));

// static Type parseOption(const llvm::cl::opt<std::string> &option) {
//   auto opt = pmlc::util::symbolizeDataType(option);
//   if (!opt.hasValue()) {
//     throw std::runtime_error(
//         llvm::formatv("Invalid runtime option: '{0}'", option));
//   }
//   return opt.getValue();
// }

struct ConstantTypesPass : public OperationPass<ConstantTypesPass> {
  ConstantTypesPass() = default;
  ConstantTypesPass(Type concreteFloat, Type concreteInt)
      : concreteFloat(concreteFloat), concreteInt(concreteInt) {}

  void runOnOperation() final;
  void notifyPassFailure() { signalPassFailure(); }

  Type concreteFloat;
  Type concreteInt;
  // DataType floatx = parseOption(clFloatxOption);
  // DataType intx = parseOption(clIntxOption);
};

bool isUnsignedInteger(Type type) {
  if (auto intTy = type.dyn_cast<IntegerType>()) {
    return intTy.isUnsigned();
  }
  return false;
}

struct ConstantTypesRewriter : public OpRewritePattern<ScalarConstantOp> {
  ConstantTypesRewriter(MLIRContext *context, ConstantTypesPass *pass,
                        Type concreteFloat, Type concreteInt)
      : OpRewritePattern<ScalarConstantOp>(context), pass(pass),
        concreteFloat(concreteFloat), concreteInt(concreteInt) {}

  LogicalResult matchAndRewrite(ScalarConstantOp constOp,
                                PatternRewriter &rewriter) const override {
    IVLOG(3, "ConstantTypesPass::matchAndRewrite");

    auto type = constOp.getType();
    auto shape = eltwise::getRankedTensorType(type).getShape();

    auto floatAttr = constOp.getFloatAttr();
    if (floatAttr && concreteFloat) {
      double value = floatAttr.getValueAsDouble();
      auto tensorType = RankedTensorType::get(shape, concreteFloat);
      if (tensorType == type) {
        return failure();
      }
      rewriter.replaceOpWithNewOp<ScalarConstantOp>(constOp, concreteFloat,
                                                    value);
      return success();
    }

    auto intAttr = constOp.getIntAttr();
    if (intAttr && concreteInt) {
      int64_t value = intAttr.getInt();
      if (isUnsignedInteger(concreteInt) && (value < 0)) {
        pass->notifyPassFailure();
        return constOp.emitOpError("Invalid Type for negative constant");
      }
      auto tensorType = RankedTensorType::get(shape, concreteInt);
      if (tensorType == type) {
        return failure();
      }
      rewriter.replaceOpWithNewOp<ScalarConstantOp>(constOp, concreteInt,
                                                    value);
      return success();
    }

    return failure();
  }

  ConstantTypesPass *pass;
  Type concreteFloat;
  Type concreteInt;
};

void ConstantTypesPass::runOnOperation() {
  OwningRewritePatternList patterns;
  patterns.insert<ConstantTypesRewriter>(&getContext(), this, concreteFloat,
                                         concreteInt);
  applyPatternsGreedily(getOperation()->getRegions(), patterns);
}

std::unique_ptr<mlir::Pass> createConstantTypesPass(Type concreteFloat,
                                                    Type concreteInt) {
  return std::make_unique<ConstantTypesPass>(concreteFloat, concreteInt);
}

static mlir::PassRegistration<ConstantTypesPass>
    pass("tile-constant-types", "Set constant types precision");

} // namespace pmlc::dialect::tile
