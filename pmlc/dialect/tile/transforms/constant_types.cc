// Copyright 2020 Intel Corporation

#include <memory>

#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/eltwise/ir/util.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/pass_detail.h"
#include "pmlc/util/enums.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::tile {

using eltwise::ScalarConstantOp;

namespace {
struct ConstantTypesPass : public ConstantTypesBase<ConstantTypesPass> {
  ConstantTypesPass() {}

  ConstantTypesPass(Type floatType, Type integerType)
      : floatType(floatType), integerType(integerType) {}

  void runOnFunction() final;

  void notifyPassFailure() { signalPassFailure(); }

  Type floatType;
  Type integerType;
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

void ConstantTypesPass::runOnFunction() {
  auto func = getFunction();
  auto *context = &getContext();

  if (!floatType) {
    IVLOG(2, "parse floatKind: " << floatKind);
    floatType = llvm::StringSwitch<Type>(floatKind)
                    .Case("f16", FloatType::getF16(context))
                    .Case("f32", FloatType::getF32(context))
                    .Case("f64", FloatType::getF64(context));
    IVLOG(2, "floatType: " << debugString(floatType));
  }

  if (!integerType) {
    IVLOG(2, "parse integerKind: " << integerKind);
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
    IVLOG(2, "integerType: " << debugString(integerType));
  }

  OwningRewritePatternList patterns;
  patterns.insert<ConstantTypesRewriter>(context, this, floatType, integerType);
  applyPatternsAndFoldGreedily(func, patterns);
}

} // namespace

std::unique_ptr<Pass> createConstantTypesPass() {
  return std::make_unique<ConstantTypesPass>();
}

std::unique_ptr<Pass> createConstantTypesPass(Type floatType,
                                              Type integerType) {
  return std::make_unique<ConstantTypesPass>(floatType, integerType);
}

} // namespace pmlc::dialect::tile
