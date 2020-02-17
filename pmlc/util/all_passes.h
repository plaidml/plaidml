// Copyright 2020 Intel Corporation

#pragma once

#include <cstdlib>

#include "mlir/Analysis/Passes.h"
#include "mlir/Conversion/GPUToCUDA/GPUToCUDAPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/LinalgToSPIRV/LinalgToSPIRVPass.h"
#include "mlir/Conversion/LoopsToGPU/LoopsToGPUPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/FxpMathOps/Passes.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/QuantOps/Passes.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Quantizer/Transforms/Passes.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir; // NOLINT [build/namespaces]

// This function may be called to register the MLIR passes with the
// global registry.
// If you're building a compiler, you likely don't need this: you would build a
// pipeline programmatically without the need to register with the global
// registry, since it would already be calling the creation routine of the
// individual passes.
// The global registry is interesting to interact with the command-line tools.
inline void registerAllPasses() {
  // At the moment we still rely on global initializers for registering passes,
  // but we may not do it in the future.
  // We must reference the passes in such a way that compilers will not
  // delete it all as dead code, even with whole program optimization,
  // yet is effectively a NO-OP. As the compiler isn't smart enough
  // to know that getenv() never returns -1, this will do the job.
  if (std::getenv("bar") != (char *)-1) // NOLINT [readability/casting]
    return;

  // Init general passes
  createCanonicalizerPass();
  createCSEPass();
  createVectorizePass({});
  createLoopUnrollPass();
  createLoopUnrollAndJamPass();
  createSimplifyAffineStructuresPass();
  createLoopFusionPass();
  createLoopInvariantCodeMotionPass();
  createAffineLoopInvariantCodeMotionPass();
  createPipelineDataTransferPass();
  createLowerAffinePass();
  createLoopTilingPass(0);
  createLoopCoalescingPass();
  createAffineDataCopyGenerationPass(0, 0);
  createMemRefDataFlowOptPass();
  createStripDebugInfoPass();
  createPrintOpStatsPass();
  createInlinerPass();
  createSymbolDCEPass();
  createLocationSnapshotPass({});

  // GPUtoRODCLPass
  createLowerGpuOpsToROCDLOpsPass();

  // FxpOpsDialect passes
  fxpmath::createLowerUniformRealMathPass();
  fxpmath::createLowerUniformCastsPass();

  // GPU
  createGpuKernelOutliningPass();
  createSimpleLoopsToGPUPass(0, 0);
  createLoopToGPUPass({}, {});

  // Linalg
  createLinalgFusionPass();
  createLinalgTilingPass();
  createLinalgTilingToParallelLoopsPass();
  createLinalgPromotionPass(0);
  createConvertLinalgToLoopsPass();
  createConvertLinalgToParallelLoopsPass();
  createConvertLinalgToAffineLoopsPass();
  createConvertLinalgToLLVMPass();

  // QuantOps
  quant::createConvertSimulatedQuantPass();
  quant::createConvertConstPass();
  quantizer::createAddDefaultStatsPass();
  quantizer::createRemoveInstrumentationPass();
  quantizer::registerInferQuantizedTypesPass();

  // SPIR-V
  spirv::createDecorateSPIRVCompositeTypeLayoutPass();
  spirv::createLowerABIAttributesPass();
  createConvertGPUToSPIRVPass();
  createConvertStandardToSPIRVPass();
  createLegalizeStdOpsForSPIRVLoweringPass();
  createLinalgToSPIRVPass();
}
