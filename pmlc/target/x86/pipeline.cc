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
#include "pmlc/dialect/xsmm/ir/ops.h"
#include "pmlc/target/x86/heatmap.h"
#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/target/x86/passes.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace pxa = dialect::pxa;
namespace xsmm = dialect::xsmm;

namespace {

struct LowerPXAToAffinePass
    : public ConvertPXAToAffineBase<LowerPXAToAffinePass> {
  void runOnOperation() final {
    auto &ctx = getContext();
    conversion::pxa_to_affine::PXAToAffineConversionTarget target(ctx);
    target.addLegalDialect<xsmm::XSMMDialect>();

    OwningRewritePatternList patterns;
    populatePXAToAffineConversionPatterns(patterns, &ctx);
    conversion::pxa_to_affine::populatePXAToAffineConversionPatterns(patterns,
                                                                     &ctx);

    if (failed(applyPartialConversion(getOperation(), target, patterns,
                                      nullptr))) {
      getOperation().dump();
      emitError(UnknownLoc::get(&ctx), "Error lowering pxa -> affine\n");
      signalPassFailure();
    }
  }
};

struct ConvertToLLVMPass
    : public PassWrapper<ConvertToLLVMPass, OperationPass<ModuleOp>> {
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
    populateXSMMToLLVMConversionPatterns(typeConverter, patterns);
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    conversion::stdx_to_llvm::populateStdXToLLVMConversionPatterns(
        typeConverter, patterns);

    LLVMConversionTarget target(*context);
    if (failed(applyPartialConversion(module, target, patterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

// NOTE: the stencil pass uses row-major ordering, the heatmap is
// specified in column-major ordering.
static pxa::StencilCost heatmapCostTransposed(ArrayRef<int64_t> tile) {
  return heatmapCost(ArrayRef<int64_t>{tile[1], tile[0], tile[2]});
}

std::unique_ptr<Pass> createXSMMStencilPass() {
  auto numThreads = std::thread::hardware_concurrency();
  return pxa::createStencilGEMMPass(numThreads, heatmapCostTransposed);
}

std::unique_ptr<Pass> createLowerPXAToAffinePass() {
  return std::make_unique<LowerPXAToAffinePass>();
}

std::unique_ptr<Pass> createLowerToLLVMPass() {
  return std::make_unique<ConvertToLLVMPass>();
}

static void addToPipeline(OpPassManager &pm) {
  pm.addPass(pmlc::dialect::tile::createComputeBoundsPass());
  pm.addPass(pmlc::dialect::tile::createPadPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(pmlc::conversion::tile_to_pxa::createLowerTileToPXAPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(
      pxa::createStencilGEMMPass(/*numThreads=*/1, heatmapCostTransposed));

  // FIXME: these passes cause test failures (correctness or otherwise)
  // pm.addPass(pxa::createFusionPass());
  // pm.addPass(createCanonicalizerPass());
  // pm.addPass(pxa::createMemRefDataFlowOptPass());
  // pm.addPass(createCanonicalizerPass());
  pm.addPass(pxa::createLocalizePass());
  pm.addPass(pxa::createResizeTmpsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createLowerPXAToAffinePass());
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerToCFGPass());
  pm.addPass(createBufferPlacementPass());
  if (pmlc::util::getEnvVar("PLAIDML_BOUNDS_CHECK") == "1") {
    pm.addPass(pmlc::dialect::stdx::createBoundsCheckPass());
  }

  pm.addPass(createLowerToLLVMPass());
  pm.addPass(createTraceLinkingPass());
}

void registerPassPipeline() {
  static PassPipelineRegistration<> passPipelineReg(
      "target-cpu", "Target pipeline for CPU", addToPipeline);
  static compiler::TargetRegistration targetReg("llvm_cpu", addToPipeline);
}

} // namespace pmlc::target::x86
