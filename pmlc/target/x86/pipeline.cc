// Copyright 2020 Intel Corporation

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/pxa_to_affine/pxa_to_affine.h"
#include "pmlc/conversion/stdx_to_llvm/stdx_to_llvm.h"
#include "pmlc/conversion/tile_to_pxa/tile_to_pxa.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/target/x86/heatmap.h"
#include "pmlc/target/x86/trace_linking.h"
#include "pmlc/target/x86/xsmm_lowering.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::pxa {

struct StencilPass;
static mlir::PassRegistration<StencilPass> xsmmStencilPass(
    "affine-stencil-xsmm",
    "Find a tiling for extracting 'micro' GEMMs suitable for XSMM.", []() {
      auto numThreads = std::thread::hardware_concurrency();
      return createStencilPass(numThreads, target::x86::heatmapCost);
    });

} // namespace pmlc::dialect::pxa

namespace pmlc::target::x86 {

namespace {

struct ConvertToLLVMPass : public ModulePass<ConvertToLLVMPass> {
  void runOnModule() override {
    auto module = getModule();
    auto *context = module.getContext();

    LLVMTypeConverterCustomization customs;
    customs.funcArgConverter = barePtrFuncArgTypeConverter;
    LLVMTypeConverter typeConverter(&getContext(), customs);

    OwningRewritePatternList patterns;
    populateXSMMConversionPatterns(patterns, context);
    populateAffineToStdConversionPatterns(patterns, context);
    populateLoopToStdConversionPatterns(patterns, context);
    populateStdToLLVMBarePtrConversionPatterns(typeConverter, patterns);
    conversion::stdx_to_llvm::populateStdXToLLVMConversionPatterns(
        typeConverter, patterns);

    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    if (failed(
            applyPartialConversion(module, target, patterns, &typeConverter))) {
      signalPassFailure();
    }
  }

  static std::unique_ptr<OpPassBase<ModuleOp>> create() {
    return std::make_unique<ConvertToLLVMPass>();
  }
};

void addToPipeline(OpPassManager &pm) {
  pm.addPass(pmlc::dialect::tile::createComputeBoundsPass());
  pm.addPass(pmlc::dialect::tile::createPadPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  pm.addPass(pmlc::conversion::tile_to_pxa::createLowerTileToPXAPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  // TODO: do optimizations here

  pm.addPass(conversion::pxa_to_affine::createLowerPXAToAffinePass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  pm.addPass(createLowerAffinePass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  pm.addPass(ConvertToLLVMPass::create());
  pm.addPass(createTraceLinkingPass());
}

static PassPipelineRegistration<>
    passPipelineReg("target-cpu", "Target pipeline for CPU", addToPipeline);
static compiler::TargetRegistration targetReg("llvm_cpu", addToPipeline);

} // namespace

} // namespace pmlc::target::x86
