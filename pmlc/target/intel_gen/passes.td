#ifndef __PMLC_TARGET_INTEL_GEN_PASSES__
#define __PMLC_TARGET_INTEL_GEN_PASSES__

include "mlir/Pass/PassBase.td"

def ConvertSimpleSCFToGPU : FunctionPass<"convert-scf-for-to-gpu"> {
  let summary = "Convert top-level loops to GPU kernels";
  let constructor = "mlir::createSimpleSCFToGPUPass()";
}

#endif // __PMLC_TARGET_INTEL_GEN_PASSES__
