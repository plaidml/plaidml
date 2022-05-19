// Copyright 2021, Intel Corporation

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "pmlc/conversion/linalg_to_pxa/pass_detail.h"
#include "llvm/Support/Debug.h"

namespace pmlc::conversion::linalg_to_pxa {

using namespace mlir; // NOLINT

namespace {

static void setupTypeConversion(ConversionTarget &target,
                                TypeConverter &typeConverter) {
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion([](RankedTensorType type) -> MemRefType {
    return MemRefType::get(type.getShape(), type.getElementType());
  });
  typeConverter.addTargetMaterialization([](OpBuilder &builder, MemRefType type,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    assert(inputs.size() == 1 && "expects one input only");
    assert(inputs[0].getType().isa<RankedTensorType>() && "expect a tensor");
    return builder.create<bufferization::ToMemrefOp>(loc, type, inputs[0]);
  });
  auto sourceMaterialization = [](OpBuilder &builder, RankedTensorType type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1 && "expects one input only");
    assert(inputs[0].getType().isa<MemRefType>() && "expect a memref");
    return builder.create<bufferization::ToTensorOp>(loc, type, inputs[0]);
  };
  typeConverter.addSourceMaterialization(sourceMaterialization);
  typeConverter.addArgumentMaterialization(sourceMaterialization);
}

struct GenericOpConversion : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    return failure();
  };
};

void populateLinalgToPXAPattern(RewritePatternSet &patterns,
                                TypeConverter &converter) {
  patterns.add<GenericOpConversion>(converter, patterns.getContext());
}

void performLinalgTransforms(ModuleOp op) {
  RewritePatternSet patterns(op.getContext());
  linalg::populateLinalgNamedOpsGeneralizationPatterns(patterns);
  populateLinalgTensorCollapseOpGeneralizationPatterns(patterns);
  populateLinalgTensorExpandOpGeneralizationPatterns(patterns);
  patterns.add<linalg::PadOpTransformationPattern>(op.getContext());
  (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
}

struct ConvertLinalgToPXA : public ConvertLinalgToPXABase<ConvertLinalgToPXA> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // generalize linalg.
    performLinalgTransforms(module);

    llvm::errs() << "Module ---->\n";
    module.dump();
    llvm::errs() << "\n----------\n";

    RewritePatternSet patterns(&getContext());
    TypeConverter typeConverter;
    ConversionTarget target(getContext());
    setupTypeConversion(target, typeConverter);

    target.addIllegalDialect<linalg::LinalgDialect>();

    // If all result types are legal, and all block arguments are legal, then
    // all types in the program are legal.
    //
    // We also check that the operand types are legal to avoid creating invalid
    // IR. For example, this prevents the patterns from updating
    // the types of the operands to a return op without updating the enclosing
    // function.
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    populateLinalgToPXAPattern(patterns, typeConverter);

    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertLinalgToPXAPass() {
  return std::make_unique<ConvertLinalgToPXA>();
}

} // namespace pmlc::conversion::linalg_to_pxa
