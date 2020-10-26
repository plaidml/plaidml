// Copyright 2020, Intel Corporation
#ifndef __PMLC_CONVERSION_COMP_TO_LLVM_PASSES__
#define __PMLC_CONVERSION_COMP_TO_LLVM_PASSES__

include "mlir/Pass/PassBase.td"

def ConvertCompToOcl : Pass<"pmlc-convert-comp-to-ocl", "mlir::ModuleOp"> {
  let summary = "Convert comp operations to OpenCL runtime calls";
  let constructor = "pmlc::conversion::comp_to_llvm::createConvertCompToOclPass()";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::StandardOpsDialect"
  ];
}

def ConvertCompToVulkanCall : Pass<"pmlc-convert-comp-to-vulkan", "mlir::ModuleOp"> {
  let summary = "Convert comp operations to Vulkan runtime calls";
  let constructor = "pmlc::conversion::comp_to_llvm::createConvertCompToVulkanPass()";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::StandardOpsDialect"
  ];
}

#endif // __PMLC_CONVERSION_COMP_TO_LLVM_PASSES__
