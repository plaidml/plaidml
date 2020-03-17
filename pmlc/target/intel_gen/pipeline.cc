// Copyright 2019, Intel Corporation

#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/LoopsToGPU/LoopsToGPUPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"

#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/gpu/kernel_outlining.h"
#include "pmlc/conversion/gpu/lowering.h"
#include "pmlc/conversion/pxa_to_affine/pxa_to_affine.h"
#include "pmlc/conversion/tile_to_pxa/tile_to_pxa.h"
#include "pmlc/dialect/tile/transforms/passes.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::intel_gen {

namespace {

void addToPipeline(OpPassManager &pm) {
  pm.addPass(dialect::tile::createComputeBoundsPass());
  // pm.addPass(dialect::tile::createPadPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(conversion::tile_to_pxa::createLowerTileToPXAPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // TODO: do optimizations here

  // pm.addPass(std::make_unique<UpdateWorkGroupSizePass>(workGroupSize));
  // pm.addPass(std::make_unique<IREETileLinalgPass>());
  // pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(conversion::pxa_to_affine::createLowerPXAToAffinePass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // pm.addPass(std::make_unique<LoopsToGPUPass>());
  pm.addPass(createSimpleLoopsToGPUPass(1, 1));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(conversion::kernel_outlining::createGpuKernelOutliningPass());

  // GPU to SPIR-V.
  pm.addPass(createLegalizeStdOpsForSPIRVLoweringPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(conversion::gpu::createLegalizeGpuOpForGpuLoweringPass());
  pm.addPass(createConvertGPUToSPIRVPass());
  // pm.addPass(std::make_unique<IREEGPUToSPIRVPass>());

  // SPIR-V passes for lowering attributes.
  // pm.addNestedPass<spirv::ModuleOp>(spirv::createLowerABIAttributesPass());
  // pm.addNestedPass<spirv::ModuleOp>(createCanonicalizerPass());
  // pm.addNestedPass<spirv::ModuleOp>(createCSEPass());

  pm.addPass(spirv::createLowerABIAttributesPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // GPU to Vulkan.
  // pm.addPass(
  //     conversion::gpu::createConvertGpuLaunchFuncToVulkanCallsPass());
  // pm.addPass(createLowerToLLVMPass());
}

static PassPipelineRegistration<>
    passPipelineReg("target-intel_gen", "Target pipeline for Intel GEN iGPUs",
                    addToPipeline);
static compiler::TargetRegistration targetReg("intel_gen", addToPipeline);

} // namespace

} // namespace pmlc::target::intel_gen
