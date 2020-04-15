// Copyright 2020 Intel Corporation

#pragma once

#include <cstdlib>

#include "mlir/Conversion/AVX512ToLLVM/ConvertAVX512ToLLVM.h"
#include "mlir/Conversion/GPUToCUDA/GPUToCUDAPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/LinalgToSPIRV/LinalgToSPIRVPass.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/LoopsToGPU/LoopsToGPUPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/LoopOps/Passes.h"
#include "mlir/Dialect/Quant/Passes.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/ViewOpGraph.h"
#include "mlir/Transforms/ViewRegionGraph.h"

#include "pmlc/conversion/pxa_to_affine/passes.h"
#include "pmlc/conversion/stdx_to_llvm/passes.h"
#include "pmlc/conversion/tile_to_pxa/passes.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/stdx/transforms/passes.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/target/x86/passes.h"

namespace mlir {

// This function may be called to register the MLIR passes with the
// global registry.
// If you're building a compiler, you likely don't need this: you would build a
// pipeline programmatically without the need to register with the global
// registry, since it would already be calling the creation routine of the
// individual passes.
// The global registry is interesting to interact with the command-line tools.
inline void registerAllPasses() {
  //
  // MLIR Core
  //

  // Init general passes
#define GEN_PASS_REGISTRATION
#include "mlir/Transforms/Passes.h.inc"

  // Conversion passes
#define GEN_PASS_REGISTRATION
#include "mlir/Conversion/Passes.h.inc"

  // Affine
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Affine/Passes.h.inc"

  // GPU
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/GPU/Passes.h.inc"

  // Linalg
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Linalg/Passes.h.inc"

  // LLVM
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"

  // Loop
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/LoopOps/Passes.h.inc"

  // Quant
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Quant/Passes.h.inc"

  // SPIR-V
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/SPIRV/Passes.h.inc"

  //
  // PMLC
  //

  // Conversion: pxa_to_affine
#define GEN_PASS_REGISTRATION
#include "pmlc/conversion/pxa_to_affine/passes.h.inc"

  // Conversion: stdx_to_llvm
#define GEN_PASS_REGISTRATION
#include "pmlc/conversion/stdx_to_llvm/passes.h.inc"

  // Conversion: tile_to_pxa
#define GEN_PASS_REGISTRATION
#include "pmlc/conversion/tile_to_pxa/passes.h.inc"

  // Dialect: pxa
#define GEN_PASS_REGISTRATION
#include "pmlc/dialect/pxa/transforms/passes.h.inc"

  // Dialect: stdx
#define GEN_PASS_REGISTRATION
#include "pmlc/dialect/stdx/transforms/passes.h.inc"

  // Dialect: tile
#define GEN_PASS_REGISTRATION
#include "pmlc/dialect/tile/transforms/passes.h.inc"

  // Target: x86
#define GEN_PASS_REGISTRATION
#include "pmlc/target/x86/passes.h.inc"
}

} // namespace mlir
