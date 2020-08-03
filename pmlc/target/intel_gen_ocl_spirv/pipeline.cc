// Copyright 2020, Intel Corporation

#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "pmlc/compiler/registry.h"

#include "pmlc/conversion/SCFToGPU/SCFToGPUPass.h"
#include "pmlc/conversion/comp/passes.h"
#include "pmlc/conversion/gpu/lowering.h"
#include "pmlc/conversion/pxa_to_affine/passes.h"
#include "pmlc/conversion/tile_to_pxa/passes.h"
#include "pmlc/dialect/comp/transforms/passes.h"
#include "pmlc/dialect/stdx/transforms/passes.h"
#include "pmlc/dialect/tile/transforms/passes.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::intel_gen_ocl_spirv {

namespace {

void addToPipeline(OpPassManager &pm) {
  // llvm::DebugFlag = true;
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

  pm.addPass(dialect::stdx::createI1StorageToI32Pass());

  pm.addPass(pmlc::conversion::scf_to_gpu::createSimpleSCFToGPUPass(1, 0));
  pm.addPass(createCanonicalizerPass());
  {
    auto outliningPass = conversion::gpu::createGpuKernelOutliningPass();
    outliningPass->initializeOptions("ocl-capabilities=true");
    pm.addPass(std::move(outliningPass));
  }

  {
    auto convertPass = conversion::comp::createConvertGpuToCompPass();
    convertPass->initializeOptions(
        "comp-execenv-runtime=1 comp-execenv-memory-space=11");
    pm.addPass(std::move(convertPass));
  }
  pm.addPass(dialect::comp::createExecEnvCoalescingPass());

  // GPU to SPIR-V.
  pm.addPass(createLegalizeStdOpsForSPIRVLoweringPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createConvertGPUToSPIRVPass());

  // SPIR-V passes for lowering attributes.
  pm.addPass(spirv::createLowerABIAttributesPass());
  pm.addPass(spirv::createUpdateVersionCapabilityExtensionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // GPU to OpenCL SPIRV.
  pm.addPass(conversion::comp::createConvertCompToOpenClPass());
  pm.addPass(conversion::gpu::createLLVMLoweringPass());
}

} // namespace

void registerPassPipeline() {
  static PassPipelineRegistration<> passPipelineReg(
      "target-intel_gen_ocl_spirv",
      "Target pipeline for Intel GEN iGPUs with OpenCL SPIRV backend",
      addToPipeline);
  static compiler::TargetRegistration targetReg("intel_gen_ocl_spirv",
                                                addToPipeline);
}

} // namespace pmlc::target::intel_gen_ocl_spirv
