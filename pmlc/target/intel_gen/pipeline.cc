// Copyright 2019, Intel Corporation

#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/LoopsToGPU/LoopsToGPUPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/pxa_to_affine/pxa_to_affine.h"

using namespace mlir;  // NOLINT[build/namespaces]
using pmlc::conversion::pxa_to_affine::createLowerPXAToAffinePass;

namespace pmlc::target::intel_gen {

static compiler::TargetRegistration pipeline("intel_gen", [](OpPassManager* pm) {
  // TODO: do optimizations here

  pm->addPass(createLowerPXAToAffinePass());
  pm->addNestedPass<FuncOp>(createCanonicalizerPass());
  pm->addNestedPass<FuncOp>(createCSEPass());

  pm->addPass(createLowerAffinePass());
  pm->addNestedPass<FuncOp>(createCanonicalizerPass());
  pm->addNestedPass<FuncOp>(createCSEPass());

  pm->addPass(createSimpleLoopsToGPUPass(1, 1));
  pm->addNestedPass<FuncOp>(createCanonicalizerPass());
  pm->addNestedPass<FuncOp>(createCSEPass());

  pm->addPass(createGpuKernelOutliningPass());
  // NOTE: canonicalizer/cse at this stage causes later passes to fail

  pm->addNestedPass<ModuleOp>(createConvertGPUToSPIRVPass({1, 1}));
  pm->addPass(createCanonicalizerPass());
  pm->addPass(createCSEPass());

  // pm->addPass(createLowerToLLVMPass(true));
});

}  // namespace pmlc::target::intel_gen
