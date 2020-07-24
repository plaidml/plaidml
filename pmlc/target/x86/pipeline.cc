// Copyright 2020 Intel Corporation

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/pxa_to_affine/passes.h"
#include "pmlc/conversion/stdx_to_llvm/passes.h"
#include "pmlc/conversion/tile_to_pxa/passes.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/stdx/transforms/passes.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/target/x86/heatmap.h"
#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/target/x86/passes.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

std::unique_ptr<Pass> createXSMMStencilPass() {
  auto numThreads = std::thread::hardware_concurrency();
  return pmlc::dialect::pxa::createXSMMStencilPass(numThreads, heatmapCost);
}

namespace {

struct ConvertToLLVMPass
    : public mlir::PassWrapper<ConvertToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
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
    populateStdToLLVMConversionPatterns(typeConverter, patterns, options);
    conversion::stdx_to_llvm::populateStdXToLLVMConversionPatterns(
        typeConverter, patterns);

    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    if (failed(applyPartialConversion(module, target, patterns))) {
      signalPassFailure();
    }
  }

  static std::unique_ptr<OperationPass<ModuleOp>> create() {
    return std::make_unique<ConvertToLLVMPass>();
  }
};

void addToPipeline(OpPassManager &pm) {
  pm.addPass(pmlc::dialect::tile::createComputeBoundsPass());
  pm.addPass(pmlc::dialect::tile::createPadPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(pmlc::conversion::tile_to_pxa::createLowerTileToPXAPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(
      pmlc::dialect::pxa::createXSMMStencilPass(/*numThreads=*/1, heatmapCost));
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createXSMMLoweringPass());

  // FIXME: these passes cause test failures (correctness or otherwise)
  // pm.addPass(pmlc::dialect::pxa::createFusionPass());
  // pm.addPass(createCanonicalizerPass());
  // pm.addPass(pmlc::dialect::pxa::createMemRefDataFlowOptPass());
  // pm.addPass(createCanonicalizerPass());
  pm.addPass(pmlc::dialect::pxa::createLocalizePass());
  pm.addPass(pmlc::dialect::pxa::createResizeTmpsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(conversion::pxa_to_affine::createLowerPXAToAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerToCFGPass());
  if (pmlc::util::getEnvVar("PLAIDML_BOUNDS_CHECK") == "1") {
    pm.addPass(pmlc::dialect::stdx::createBoundsCheckPass());
  }

  pm.addPass(createTanhLoweringPass());

  pm.addPass(ConvertToLLVMPass::create());
  pm.addPass(createTraceLinkingPass());
}

} // namespace

void registerPassPipeline() {
  static PassPipelineRegistration<> passPipelineReg(
      "target-cpu", "Target pipeline for CPU", addToPipeline);
  static compiler::TargetRegistration targetReg("llvm_cpu", addToPipeline);
}

} // namespace pmlc::target::x86
