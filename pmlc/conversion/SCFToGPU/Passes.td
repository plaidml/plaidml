//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SCF_TO_GPU_PASSES
#define MLIR_SCF_TO_GPU_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// SCFToGPU
//===----------------------------------------------------------------------===//

def ConvertSimpleSCFToGPU : FunctionPass<"convert-scf-to-gpu"> {
  let summary = "Convert top-level loops to GPU kernels";
  let constructor = "createSimpleSCFToGPUPass()";
  let dependentDialects = ["mlir::gpu::GPUDialect"];
  let options = [
    Option<"numBlockDims", "gpu-block-dims", "unsigned", /*default=*/"1u",
           "Number of GPU block dimensions for mapping">,
    Option<"numThreadDims", "gpu-thread-dims", "unsigned", /*default=*/"1u",
           "Number of GPU thread dimensions for mapping">
  ];
}

#endif // MLIR_SCF_TO_GPU_PASSES
