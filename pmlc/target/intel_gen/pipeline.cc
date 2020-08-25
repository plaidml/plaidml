// Copyright 2020, Intel Corporation

#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/compiler/registry.h"

#include "pmlc/conversion/SCFToGPU/SCFToGPUPass.h"
#include "pmlc/conversion/gpu/lowering.h"
#include "pmlc/conversion/pxa_to_affine/passes.h"
#include "pmlc/conversion/stdx_to_llvm/passes.h"
#include "pmlc/conversion/tile_to_pxa/passes.h"
#include "pmlc/dialect/pxa//transforms/passes.h"
#include "pmlc/dialect/stdx/transforms/passes.h"
#include "pmlc/dialect/tile/transforms/passes.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::intel_gen {

namespace {
// Lower std and stdx ops to llvm dialect on gpu host function
struct LowerStdAndStdxToLLVMPass
    : public PassWrapper<LowerStdAndStdxToLLVMPass, OperationPass<ModuleOp>> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = module.getContext();

    LowerToLLVMOptions options = {
        /*useBarePtrCallConv=*/false,
        /*emitCWrappers=*/true,
        /*indexBitwidth=*/kDeriveIndexBitwidthFromDataLayout,
        /*useAlignedAlloc=*/false,
    };
    LLVMTypeConverter typeConverter(context, options);

    OwningRewritePatternList patterns;
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    conversion::stdx_to_llvm::populateStdXToLLVMConversionPatterns(
        typeConverter, patterns);

    LLVMConversionTarget target(*context);
    if (failed(applyPartialConversion(module, target, patterns))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createLowerStdAndStdxToLLVMPass() {
  return std::make_unique<LowerStdAndStdxToLLVMPass>();
}

void pipelineBuilder(OpPassManager &pm) {
  pm.getContext()->getOrLoadDialect<spirv::SPIRVDialect>();

  pm.addPass(dialect::tile::createComputeBoundsPass());
  // pm.addPass(dialect::tile::createPadPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(conversion::tile_to_pxa::createLowerTileToPXAPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(pmlc::dialect::pxa::createTileAccumulatePass());

  pm.addPass(conversion::pxa_to_affine::createLowerPXAToAffinePass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(pmlc::conversion::scf_to_gpu::createSimpleSCFToGPUPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(conversion::gpu::createGpuKernelOutliningPass());

  // GPU to SPIR-V.
  pm.addPass(createLegalizeStdOpsForSPIRVLoweringPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createConvertGPUToSPIRVPass());

  // SPIR-V passes for lowering attributes.
  pm.addPass(spirv::createLowerABIAttributesPass());
  pm.addPass(spirv::createUpdateVersionCapabilityExtensionPass());

  // GPU to Vulkan.
  pm.addPass(conversion::gpu::createConvertGpuLaunchFuncToVulkanCallsPass());
  pm.addPass(createLowerStdAndStdxToLLVMPass());
}

} // namespace

static PassPipelineRegistration<>
    passPipelineReg("target-intel_gen", "Target pipeline for Intel GEN iGPUs",
                    pipelineBuilder);
static compiler::TargetRegistration targetReg("intel_gen", pipelineBuilder);

} // namespace pmlc::target::intel_gen
