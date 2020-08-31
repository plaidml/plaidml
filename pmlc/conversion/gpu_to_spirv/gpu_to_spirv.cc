// Copyright 2020 Intel Corporation

#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"

#include "pmlc/conversion/gpu_to_spirv/pass_detail.h"
#include "pmlc/conversion/gpu_to_spirv/passes.h"
#include "pmlc/dialect/stdx/ir/ops.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::conversion::gpu_to_spirv {
namespace stdx = dialect::stdx;

namespace {
/// Pass to lower to SPIRV that includes GPU, SCF, Std and Stdx dialects
struct StdxSubgroupBroadcastOpConversion
    : public SPIRVOpLowering<stdx::SubgroupBroadcastOp> {
  using SPIRVOpLowering<stdx::SubgroupBroadcastOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(stdx::SubgroupBroadcastOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto stdxType = op.getResult().getType();
    auto spirvType = typeConverter.convertType(stdxType);
    rewriter.replaceOpWithNewOp<spirv::GroupBroadcastOp>(
        op, spirvType, spirv::Scope::Subgroup, op.getOperand(0),
        op.getOperand(1));

    return success();
  }
};

template <typename StdxOpTy, typename SpirvOpTy>
struct StdxUnaryOpConversion : public SPIRVOpLowering<StdxOpTy> {
  using SPIRVOpLowering<StdxOpTy>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(StdxOpTy op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 1);
    auto dstType = op.getResult().getType();
    rewriter.replaceOpWithNewOp<SpirvOpTy>(op, dstType, operands.front());
    return success();
  }
};

struct StdxRoundOpConversion : public SPIRVOpLowering<stdx::RoundOp> {
  using SPIRVOpLowering<stdx::RoundOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(stdx::RoundOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 1);
    auto stdxType = op.getResult().getType();
    auto loc = op.getLoc();
    // Since there is no round op in SPIRV dialect right now,
    // implement round(x) by floor(x + 0.5)
    auto cstOp = rewriter.create<spirv::ConstantOp>(
        loc, stdxType, FloatAttr::get(stdxType, 0.5));
    auto addOp = rewriter.create<spirv::FAddOp>(loc, cstOp.getResult(),
                                                operands.front());
    rewriter.replaceOpWithNewOp<spirv::GLSLFloorOp>(op, stdxType,
                                                    addOp.getResult());

    return success();
  }
};

struct StdxCosHOpConversion : public SPIRVOpLowering<stdx::CosHOp> {
  using SPIRVOpLowering<stdx::CosHOp>::SPIRVOpLowering;
  LogicalResult
  matchAndRewrite(stdx::CosHOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 1);
    auto loc = op.getLoc();
    auto operand = operands.front();
    auto srcType = operand.getType();
    auto posExp = rewriter.create<spirv::GLSLExpOp>(loc, srcType, operand);
    auto negOperand = rewriter.create<spirv::FNegateOp>(loc, operand);
    auto negExp = rewriter.create<spirv::GLSLExpOp>(loc, srcType, negOperand);
    auto expSum = rewriter.create<spirv::FAddOp>(loc, posExp, negExp);
    auto cst2 = rewriter.create<spirv::ConstantOp>(
        loc, srcType, FloatAttr::get(srcType, 2.0));
    auto divOp = rewriter.create<spirv::FDivOp>(loc, expSum, cst2);
    op.getResult().replaceAllUsesWith(divOp.getResult());
    op.erase();
    return success();
  }
};

struct StdxSinHOpConversion : public SPIRVOpLowering<stdx::SinHOp> {
  using SPIRVOpLowering<stdx::SinHOp>::SPIRVOpLowering;
  LogicalResult
  matchAndRewrite(stdx::SinHOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 1);
    auto loc = op.getLoc();
    auto operand = operands.front();
    auto srcType = operand.getType();
    auto posExp = rewriter.create<spirv::GLSLExpOp>(loc, srcType, operand);
    auto negOperand = rewriter.create<spirv::FNegateOp>(loc, operand);
    auto negExp = rewriter.create<spirv::GLSLExpOp>(loc, srcType, negOperand);
    auto expSum = rewriter.create<spirv::FSubOp>(loc, posExp, negExp);
    auto cst2 = rewriter.create<spirv::ConstantOp>(
        loc, srcType, FloatAttr::get(srcType, 2.0));
    auto divOp = rewriter.create<spirv::FDivOp>(loc, expSum, cst2);
    op.getResult().replaceAllUsesWith(divOp.getResult());
    op.erase();
    return success();
  }
};

struct GPUToSPIRVCustomPass
    : public GPUToSPIRVCustomBase<GPUToSPIRVCustomPass> {
  void runOnOperation() final {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    SmallVector<Operation *, 1> kernelModules;
    OpBuilder builder(context);
    module.walk([&builder, &kernelModules](gpu::GPUModuleOp moduleOp) {
      // For each kernel module (should be only 1 for now, but that is not a
      // requirement here), clone the module for conversion because the
      // gpu.launch function still needs the kernel module.
      builder.setInsertionPoint(moduleOp.getOperation());
      kernelModules.push_back(builder.clone(*moduleOp.getOperation()));
    });

    auto targetAttr = spirv::lookupTargetEnvOrDefault(module);
    std::unique_ptr<ConversionTarget> target =
        spirv::SPIRVConversionTarget::get(targetAttr);

    SPIRVTypeConverter typeConverter(targetAttr);
    ScfToSPIRVContext scfContext;
    OwningRewritePatternList patterns;
    populateGPUToSPIRVPatterns(context, typeConverter, patterns);
    populateSCFToSPIRVPatterns(context, typeConverter, scfContext, patterns);
    populateStandardToSPIRVPatterns(context, typeConverter, patterns);
    populateStdxToSPIRVPatterns(context, typeConverter, patterns);
    if (spirv::getMemoryModel(targetAttr) == spirv::MemoryModel::GLSL450)
      populateStdxToSPIRVGLSLPatterns(context, typeConverter, patterns);

    if (failed(applyFullConversion(kernelModules, *target, patterns)))
      return signalPassFailure();
  }
};
} // namespace

void populateStdxToSPIRVPatterns(MLIRContext *context,
                                 SPIRVTypeConverter &typeConverter,
                                 OwningRewritePatternList &patterns) {
  patterns.insert<StdxSubgroupBroadcastOpConversion>(context, typeConverter);
}

void populateStdxToSPIRVGLSLPatterns(MLIRContext *context,
                                     SPIRVTypeConverter &typeConverter,
                                     OwningRewritePatternList &patterns) {
  patterns.insert<StdxRoundOpConversion,
                  StdxUnaryOpConversion<stdx::FloorOp, spirv::GLSLFloorOp>,
                  StdxUnaryOpConversion<stdx::TanOp, spirv::GLSLTanOp>,
                  StdxCosHOpConversion, StdxSinHOpConversion>(context,
                                                              typeConverter);
}

std::unique_ptr<Pass> createGPUToSPIRVCustomPass() {
  return std::make_unique<GPUToSPIRVCustomPass>();
}

} // namespace pmlc::conversion::gpu_to_spirv
