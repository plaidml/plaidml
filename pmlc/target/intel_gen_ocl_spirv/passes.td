// Copyright 2020, Intel Corporation
#ifndef __PMLC_TARGET_INTEL_GEN__OCL_SPIRV_PASSES__
#define __PMLC_TARGET_INTEL_GEN__OCL_SPIRV_PASSES__

include "mlir/Pass/PassBase.td"

def IntelGenOclAddSpirvTarget : Pass<"intel-gen-ocl-add-spirv-target", "mlir::ModuleOp"> {
  let summary = "Add spirv target environment attributes";
  let constructor = "pmlc::target::intel_gen_ocl_spirv::createAddSpirvTargetPass()";
  let dependentDialects = ["mlir::spirv::SPIRVDialect"];
}

#endif // __PMLC_TARGET_INTEL_GEN__OCL_SPIRV_PASSES__
