#ifndef __PMLC_CONVERSION_GPU_TO_COMP_PASSES__
#define __PMLC_CONVERSION_GPU_TO_COMP_PASSES__

include "mlir/Pass/PassBase.td"

def ConvertGpuToComp : Pass<"pmlc-convert-gpu-to-comp", "mlir::ModuleOp"> {
  let summary = "Convert gpu operations to comp operations";
  let constructor = "pmlc::conversion::comp::createConvertGpuToCompPass()";
  let options = [
    Option<"execEnvRuntime", "comp-execenv-runtime", "unsigned",
           /*default=*/"0", "Runtime to use for gpu kernels.">,
    Option<"execEnvMemorySpace", "comp-execenv-memory-space", "unsigned",
           /*default=*/"0", "Memory space to use for data transferred to device.">
  ];
}

def ConvertCompToOpenCl : Pass<"pmlc-convert-comp-to-opencl", "mlir::ModuleOp"> {
  let summary = "TODO";
  let constructor = "pmlc::conversion::comp::createConvertCompToOpenClPass()";
}

#endif // __PMLC_CONVERSION_GPU_TO_VULKAN_PASSES__
