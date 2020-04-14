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
using mlir::FloatType;
using mlir::IntegerType;
using mlir::LogicalResult;
using mlir::OperationPass;
using mlir::OpRewritePattern;
using mlir::OwningRewritePatternList;
using mlir::PassWrapper;
using mlir::PatternRewriter;
using mlir::success;
using mlir::Type;

using eltwise::ScalarConstantOp;

struct ConstantTypesPass
    : public PassWrapper<ConstantTypesPass, OperationPass<void>> {
  ConstantTypesPass() {}

  ConstantTypesPass(const ConstantTypesPass &rhs) {
    floatType = rhs.floatType;
    integerType = rhs.integerType;
    floatKind = rhs.floatKind.getValue();
    integerKind = rhs.integerKind.getValue();
  }

  ConstantTypesPass(Type floatType, Type integerType)
      : floatType(floatType), integerType(integerType) {}

  void runOnOperation() final;

  void notifyPassFailure() { signalPassFailure(); }

  llvm::Optional<Type> floatType;
  llvm::Optional<Type> integerType;

  Option<std::string> floatKind{
      *this, "floatx", llvm::cl::desc("set floating-point constant precision"),
      llvm::cl::init("f32")};
  Option<std::string> integerKind{
      *this, "intx", llvm::cl::desc("set integer constant precision"),
      llvm::cl::init("si32")};
};

struct ConstantTypesRewriter : public OpRewritePattern<ScalarConstantOp> {
  ConstantTypesRewriter(MLIRContext *context, ConstantTypesPass *pass,
                        Type floatType, Type integerType)
      : OpRewritePattern<ScalarConstantOp>(context), pass(pass),
        floatType(floatType), integerType(integerType) {}

  LogicalResult matchAndRewrite(ScalarConstantOp constOp,
                                PatternRewriter &rewriter) const override {
    auto type = constOp.getType();
    auto shape = eltwise::getRankedTensorType(type).getShape();

    auto floatAttr = constOp.getFloatAttr();
    if (floatAttr && floatType) {
      double value = floatAttr.getValueAsDouble();
      auto tensorType = RankedTensorType::get(shape, floatType);
      if (tensorType == type) {
        return failure();
      }
      rewriter.replaceOpWithNewOp<ScalarConstantOp>(constOp, floatType, value);
      return success();
    }

    auto intAttr = constOp.getIntAttr();
    if (intAttr && integerType) {
      Type localType = integerType;
      int64_t value = intAttr.getInt();
      if (localType.isUnsignedInteger() && value < 0) {
        pass->notifyPassFailure();
        return constOp.emitOpError("Invalid Type for negative constant");
      }
      auto tensorType = RankedTensorType::get(shape, integerType);
      if (tensorType == type) {
        return failure();
      }
      rewriter.replaceOpWithNewOp<ScalarConstantOp>(constOp, integerType,
                                                    value);
      return success();
    }

    return failure();
  }

  ConstantTypesPass *pass;
  Type floatType;
  Type integerType;
};

void ConstantTypesPass::runOnOperation() {
  auto *context = &getContext();

  if (!floatType) {
    IVLOG(1, "parse floatKind: " << floatKind);
    floatType = llvm::StringSwitch<Type>(floatKind)
                    .Case("f16", FloatType::getF16(context))
                    .Case("f32", FloatType::getF32(context))
                    .Case("f64", FloatType::getF64(context));
    IVLOG(1, "floatType: " << mlir::debugString(*floatType));
  }

  if (!integerType) {
    IVLOG(1, "parse integerKind: " << integerKind);
    integerType =
        llvm::StringSwitch<Type>(integerKind)
            .Case("si8", IntegerType::get(8, IntegerType::Signed, context))
            .Case("ui8", IntegerType::get(8, IntegerType::Unsigned, context))
            .Case("si16", IntegerType::get(16, IntegerType::Signed, context))
            .Case("ui16", IntegerType::get(16, IntegerType::Unsigned, context))
            .Case("si32", IntegerType::get(32, IntegerType::Signed, context))
            .Case("ui32", IntegerType::get(32, IntegerType::Unsigned, context))
            .Case("si64", IntegerType::get(64, IntegerType::Signed, context))
            .Case("ui64", IntegerType::get(64, IntegerType::Unsigned, context));
    IVLOG(1, "integerType: " << mlir::debugString(*integerType));
  }

  OwningRewritePatternList patterns;
  patterns.insert<ConstantTypesRewriter>(&getContext(), this, *floatType,
                                         *integerType);
  applyPatternsAndFoldGreedily(getOperation()->getRegions(), patterns);
}

std::unique_ptr<mlir::Pass> createConstantTypesPass(Type floatType,
                                                    Type integerType) {
  return std::make_unique<ConstantTypesPass>(floatType, integerType);
}

static mlir::PassRegistration<ConstantTypesPass>
    pass("tile-constant-types", "Set constant types precision");

} // namespace pmlc::dialect::tile
