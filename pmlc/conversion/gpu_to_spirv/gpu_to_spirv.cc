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

std::unique_ptr<Pass> createGPUToSPIRVCustomPass() {
  return std::make_unique<GPUToSPIRVCustomPass>();
}

} // namespace pmlc::conversion::gpu_to_spirv
