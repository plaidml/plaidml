#ifndef __PMLC_CONVERSION_GPU_TO_VULKAN_PASSES__
#define __PMLC_CONVERSION_GPU_TO_VULKAN_PASSES__

include "mlir/Pass/PassBase.td"

def ConvertGpuLaunchFuncToVulkanCalls : Pass<"pmlc-convert-gpu-to-vulkan", "mlir::ModuleOp"> {
  let summary = "Convert gpu.launch_func op to Vulkan runtime calls";
  let constructor = "pmlc::conversion::gpu::createConvertGpuLaunchFuncToVulkanCallsPass()";
}

def GpuKernelOutliningPass : Pass<"pmlc-kernel_outlining", "mlir::ModuleOp"> {
  let summary = "Outline gpu.launch bodies to kernel functions";
  let constructor = "pmlc::conversion::gpu::createGpuKernelOutliningPass()";
  let options = [
    Option<"oclCapabilities", "ocl-capabilities", "bool", /*default=*/"false",
           "Flag to generate OpenCL flavor SPIRV capabilities instead of default Vulkan ones">,
  ];
}

#endif // __PMLC_CONVERSION_GPU_TO_VULKAN_PASSES__
