// Copyright 2020, Intel Corporation

#include "llvm/Support/FormatVariadic.h"

#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/comp_to_llvm/passes.h"
#include "pmlc/conversion/gpu/lowering.h"
#include "pmlc/conversion/gpu_to_comp/passes.h"
#include "pmlc/conversion/gpu_to_spirv/passes.h"
#include "pmlc/conversion/pxa_to_affine/passes.h"
#include "pmlc/conversion/stdx_to_llvm/passes.h"
#include "pmlc/conversion/tile_to_pxa/passes.h"
#include "pmlc/dialect/comp/ir/types.h"
#include "pmlc/dialect/comp/transforms/passes.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/stdx/transforms/passes.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/target/intel_gen/passes.h"
#include "pmlc/target/intel_gen_ocl_spirv/passes.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::intel_gen_ocl_spirv {

namespace comp = pmlc::dialect::comp;
namespace pxa = pmlc::dialect::pxa;

void pipelineBuilder(OpPassManager &pm) {
  // Bound + pad initial tile code
  pm.addPass(dialect::tile::createComputeBoundsPass());
  pm.addPass(dialect::tile::createPadRangesPass());
  pm.addPass(dialect::tile::createPadConstraintsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Lower to PXA
  pm.addPass(conversion::tile_to_pxa::createLowerTileToPXAPass());
  pm.addPass(pmlc::dialect::pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Do subgroup or accumulation
  pm.addPass(pmlc::dialect::pxa::createSubgroupsPass());
  pm.addPass(pmlc::dialect::pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Do tiled fusion
  pm.addPass(pxa::createFusionPass(/*memoryActivityThreshold=*/0,
                                   /*exactlyMatch=*/false, /*tiledFusion=*/true,
                                   /*loopDepth=*/3));
  pm.addPass(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(pxa::createMemRefDataFlowOptPass(/*onlyParallelNested=*/true));
  pm.addPass(createCanonicalizerPass());
  // TODO: parametrize localize pass depending on memory size and HW caps
  pm.addPass(pxa::createLocalizePass());
  pm.addPass(pxa::createResizeTmpsPass(/*onlyParallelNested=*/true));
  pm.addPass(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Assign GPU blocks + threads to outermost loop
  pm.addPass(pmlc::dialect::pxa::createGPUThreadPass(/*maxThreads=*/64));
  pm.addPass(pmlc::dialect::pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Data layout optimization.
  pm.addPass(createIntelGenOclReorderLayoutsPass(/*maxThreads=*/64,
                                                 /*allowReorder=*/false));
  pm.addPass(pxa::createSimplifyWithConstraintsPass());
  pm.addPass(pmlc::dialect::pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // TODO: uncomment this pass after llvm upstream update with dynamic vec ops
  /*pm.addPass(pxa::createVectorizeMemPass());
  pm.addPass(pmlc::dialect::pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());*/

  // Lower out of PXA memory semantics
  pm.addPass(pmlc::target::intel_gen::createLowerPXAToAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createAffineLoopInvariantCodeMotionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Unroll affine.for loops.
  pm.addPass(createLoopUnrollPass(
      /*unrollFactor=*/256,
      /*unrollUpToFactor=*/true));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Pack dims
  pm.addPass(pmlc::target::intel_gen::createAffineIndexPackPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Do a custom version of lower-affine which also set GPU mappings
  pm.addPass(pmlc::target::intel_gen::createIntelGenLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Fix booleans
  pm.addPass(dialect::stdx::createI1StorageToI32Pass());

  // Devectorize
  pm.addPass(pmlc::target::intel_gen::createSubgroupBroadcastPass(
      /*useBlockOpsr=*/true));
  pm.addPass(createCSEPass());

  // Lower mapped scf.parallel's to GPU
  pm.addPass(pmlc::target::intel_gen::createParallelLoopToGpuPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Do kernel outlining
  pm.addPass(createAddSpirvTargetPass());
  pm.addPass(conversion::gpu::createGpuKernelOutliningPass());

  // Convert GPU to comp.
  pm.addPass(pmlc::conversion::gpu_to_comp::createConvertGpuToCompPass(
      comp::ExecEnvRuntime::OpenCL, /*memorySpace=*/11));
  pm.addPass(comp::createExecEnvCoalescingPass());
  pm.addPass(comp::createMinimizeAllocationsPass());
  pm.addPass(comp::createRemoveRedundantRWPass());
  pm.addPass(comp::createRecalculateEventDepsPass(/*safeDealloc=*/false));

  // GPU to SPIR-V.
  pm.addPass(createLegalizeStdOpsForSPIRVLoweringPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(conversion::gpu_to_spirv::createGPUToSPIRVCustomPass());

  // SPIR-V passes for lowering attributes.
  pm.addPass(createSetSubgroupSizePass());
  pm.addPass(createLegalizeSpirvPass());
  pm.addPass(spirv::createLowerABIAttributesPass());
  pm.addPass(spirv::createUpdateVersionCapabilityExtensionPass());

  // Comp to LLVM - OpenCL function calls.
  pm.addPass(pmlc::conversion::comp_to_llvm::createConvertCompToOclPass());

  // Convert to LLVM code.
  pm.addPass(pmlc::target::intel_gen::createConvertStandardToLLVM());
}

static constexpr const char *kTargetName = "intel_gen_ocl_spirv";
static constexpr const char *kPassPipelineTargetName =
    "target-intel_gen_ocl_spirv";

static PassPipelineRegistration<>
    passPipelineReg(kPassPipelineTargetName,
                    "Target pipeline for Intel GEN iGPUs", pipelineBuilder);

class Target : public compiler::Target {
public:
  void buildPipeline(mlir::OpPassManager &pm) { pipelineBuilder(pm); }

  util::BufferPtr save(compiler::Program &program) {
    throw std::runtime_error(
        llvm::formatv("Target '{0}' does not have 'save' support.", kTargetName)
            .str());
  }
};

static compiler::TargetRegistration targetReg(kTargetName, []() {
  return std::make_shared<Target>();
});

} // namespace pmlc::target::intel_gen_ocl_spirv
