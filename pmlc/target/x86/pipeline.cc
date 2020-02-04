// Copyright 2019, Intel Corporation

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/pxa_to_affine/pxa_to_affine.h"
#include "pmlc/conversion/stdx_to_llvm/stdx_to_llvm.h"
#include "pmlc/target/x86/trace_linking.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace {

void addToPipeline(OpPassManager &pm) {
  // TODO: do optimizations here

  pm.addPass(conversion::pxa_to_affine::createLowerPXAToAffinePass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  pm.addPass(createLowerAffinePass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  pm.addPass(conversion::stdx_to_llvm::createLowerToLLVMPass());
  pm.addPass(createTraceLinkingPass());
}

static PassPipelineRegistration<>
    passPipelineReg("target-cpu", "Target pipeline for CPU", addToPipeline);
static compiler::TargetRegistration targetReg("llvm_cpu", addToPipeline);

} // namespace

} // namespace pmlc::target::x86
