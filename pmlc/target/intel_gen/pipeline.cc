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
#include "mlir/Dialect/Vector/VectorOps.h"
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
#include "pmlc/target/intel_gen/pass_detail.h"
#include "pmlc/target/intel_gen/passes.h"
#include "pmlc/target/x86/passes.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::intel_gen {

namespace pxa = pmlc::dialect::pxa;
namespace comp = pmlc::dialect::comp;

namespace {

struct LowerPXAToAffinePass
    : public ConvertPXAToAffineBase<LowerPXAToAffinePass> {
  void runOnOperation() final {
    auto &ctx = getContext();
    conversion::pxa_to_affine::PXAToAffineConversionTarget target(ctx);

    OwningRewritePatternList patterns;
    x86::populatePXAPrngToAffineConversionPatterns(patterns, &ctx);
    conversion::pxa_to_affine::populatePXAToAffineConversionPatterns(patterns,
                                                                     &ctx);

    if (failed(applyPartialConversion(getOperation(), target, patterns,
                                      nullptr))) {
      getOperation().emitError("Error lowering pxa -> affine\n");
      signalPassFailure();
    }
  }
};

struct ConvertStandardToLLVMPass
    : public ConvertStandardToLLVMBase<ConvertStandardToLLVMPass> {
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
    populateExpandTanhPattern(patterns, context);
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    conversion::stdx_to_llvm::populateStdXToLLVMConversionPatterns(
        typeConverter, patterns);

    LLVMConversionTarget target(*context);
    if (failed(applyPartialConversion(module, target, patterns))) {
      signalPassFailure();
    }
  }
};

struct ParallelLoopToGpuPass
    : public ConvertParallelLoopToGpuBase<ParallelLoopToGpuPass> {
  void runOnOperation() override {
    OwningRewritePatternList patterns;
    populateParallelLoopToGPUPatterns(patterns, &getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<pmlc::dialect::stdx::StdXDialect>();
    target.addLegalDialect<AffineDialect>();
    target.addLegalDialect<gpu::GPUDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalOp<vector::InsertElementOp>();
    target.addIllegalOp<scf::ParallelOp>();
    if (failed(applyPartialConversion(getOperation(), target, patterns)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertStandardToLLVM() {
  return std::make_unique<ConvertStandardToLLVMPass>();
}

std::unique_ptr<Pass> createParallelLoopToGpuPass() {
  return std::make_unique<ParallelLoopToGpuPass>();
}

std::unique_ptr<Pass> createLowerPXAToAffinePass() {
  return std::make_unique<LowerPXAToAffinePass>();
}

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

  // Lower out of PXA memory semantics
  pm.addPass(createLowerPXAToAffinePass());

  // Unroll affine.for loops.
  pm.addPass(createLoopUnrollPass(
      /*unrollFactor=*/256,
      /*unrollUpToFactor=*/true));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Pack dims
  pm.addPass(createAffineIndexPackPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Do a custom version of lower-affine which also set GPU mappings
  pm.addPass(createIntelGenLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Fix booleans
  pm.addPass(dialect::stdx::createI1StorageToI32Pass());

  // Devectorize
  pm.addPass(createSubgroupBroadcastPass());
  pm.addPass(createCSEPass());

  // Lower mapped scf.parallel's to GPU
  pm.addPass(createParallelLoopToGpuPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Do kernel outlining
  pm.addPass(conversion::gpu::createGpuKernelOutliningPass());

  // Convert GPU to comp.
  pm.addPass(pmlc::conversion::gpu_to_comp::createConvertGpuToCompPass(
      comp::ExecEnvRuntime::Vulkan, /*memorySpace=*/0));
  pm.addPass(comp::createExecEnvCoalescingPass());
  pm.addPass(comp::createMinimizeAllocationsPass());

  // GPU to SPIR-V.
  pm.addPass(createLegalizeStdOpsForSPIRVLoweringPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(conversion::gpu_to_spirv::createGPUToSPIRVCustomPass());

  // SPIR-V passes for lowering attributes.
  pm.addPass(spirv::createLowerABIAttributesPass());
  pm.addPass(spirv::createUpdateVersionCapabilityExtensionPass());

  // Comp to LLVM - Vulkan function calls.
  pm.addPass(pmlc::conversion::comp_to_llvm::createConvertCompToVulkanPass());

  // Convert Vulkan calls to LLVM code
  pm.addPass(createConvertStandardToLLVM());
}

static constexpr const char *kTargetName = "intel_gen";
static constexpr const char *kPassPipelineTargetName = "target-intel_gen";

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

} // namespace pmlc::target::intel_gen
