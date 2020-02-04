// Copyright 2019, Intel Corporation

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/pxa_to_affine/pxa_to_affine.h"
#include "pmlc/target/x86/trace_linking.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]
using pmlc::conversion::pxa_to_affine::createLowerPXAToAffinePass;

namespace pmlc::target::x86 {

namespace {

struct ConvertToLLVMPass : public ModulePass<ConvertToLLVMPass> {
  void runOnModule() override {
    auto module = getModule();
    auto *context = module.getContext();
    LLVM::ensureDistinctSuccessors(module);
    LLVMTypeConverter typeConverter(context);

    OwningRewritePatternList patterns;
    populateAffineToStdConversionPatterns(patterns, context);
    populateLoopToStdConversionPatterns(patterns, context);
    populateStdToLLVMConversionPatterns(typeConverter, patterns);

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
  // TODO: do optimizations here

  pm.addPass(createLowerPXAToAffinePass());
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
