// Copyright 2020, Intel Corporation
#ifndef __PMLC_CONVERSION_GPU_TO_COMP_PASSES__
#define __PMLC_CONVERSION_GPU_TO_COMP_PASSES__

include "mlir/Pass/PassBase.td"

def ConvertGpuToComp : Pass<"pmlc-convert-gpu-to-comp", "mlir::ModuleOp"> {
  let summary = "Convert gpu operations to comp operations";
  let description = [{
    Wraps gpu.launch_func operations inside comp.schedule_func and adds
    necessary memory transfers between host and device.
    Execution environments are created seperately for each function launch.
    Converts all kernel function signatures to use memrefs in specified
    memory space.
  }];
  let constructor = "pmlc::conversion::gpu_to_comp::createConvertGpuToCompPass()";
  let options = [
    Option<"execEnvRuntime", "comp-execenv-runtime", "unsigned",
           /*default=*/"0", "Runtime to use for gpu kernels.">,
    Option<"execEnvMemorySpace", "comp-execenv-memory-space", "unsigned",
           /*default=*/"0", "Memory space to use for data transferred to device.">
  ];
  let dependentDialects = [
    "dialect::comp::COMPDialect",
    "mlir::gpu::GPUDialect"
  ];
}

#endif // __PMLC_CONVERSION_GPU_TO_COMP_PASSES__
