// Copyright 2020, Intel Corporation
#ifndef __PMLC_GPU_TRANSFORMS__
#define __PMLC_GPU_TRANSFORMS__

include "mlir/Pass/PassBase.td"

def GpuKernelOutliningPass : Pass<"pmlc-kernel_outlining", "mlir::ModuleOp"> {
  let summary = "Outline gpu.launch bodies to kernel functions";
  let constructor = "pmlc::conversion::gpu::createGpuKernelOutliningPass()";
  let options = [
    Option<"execEnvRuntime", "comp-execenv-runtime", "unsigned",
           /*default=*/"0", "Runtime to use for gpu kernels.">,
    Option<"execEnvMemorySpace", "comp-execenv-memory-space", "unsigned",
           /*default=*/"0", "Memory space to use for data transferred to device.">
  ];

  let dependentDialects = [
    "dialect::comp::COMPDialect",
    "mlir::gpu::GPUDialect",
    "mlir::spirv::SPIRVDialect"
  ];
}

def GatherGpuLaunchFuncsPass : Pass<"pmlc-gather_gpu_launch_funcs", "mlir::ModuleOp"> {
  let summary = "Gather gpu.launch_func separated by std.alloc";
  let constructor = "pmlc::conversion::gpu::createGatherGpuLaunchFuncsPass()";
  let dependentDialects = [
    "mlir::gpu::GPUDialect",
    "mlir::spirv::SPIRVDialect"
  ];
}


#endif // __PMLC_GPU_TRANSFORMS__
