// Copyright 2020, Intel Corporation
#ifndef __PMLC_TARGET_INTEL_GEN__OCL_SPIRV_PASSES__
#define __PMLC_TARGET_INTEL_GEN__OCL_SPIRV_PASSES__

include "mlir/Pass/PassBase.td"

def IntelGenOclAddSpirvTarget : Pass<"intel-gen-ocl-add-spirv-target", "mlir::ModuleOp"> {
  let summary = "Add spirv target environment attributes";
  let constructor = "pmlc::target::intel_gen_ocl_spirv::createAddSpirvTargetPass()";
  let dependentDialects = ["mlir::spirv::SPIRVDialect"];
}

def IntelGenOclSetSubgroupSize : Pass<"intel-gen-ocl-set-subgroup-size", "mlir::spirv::ModuleOp"> {
  let summary = "Set sub-group size for each spirv module to first (x) block dimension";
  let constructor = "pmlc::target::intel_gen_ocl_spirv::createSetSubgroupSizePass()";
  let dependentDialects = ["mlir::spirv::SPIRVDialect"];
}

def IntelGenOclLegalizeSpirv : Pass<"intel-gen-ocl-legalize-spirv", "mlir::spirv::ModuleOp"> {
  let summary = "Legalize spirv for OpenCL execution environment";
  let constructor = "pmlc::target::intel_gen_ocl_spirv::createLegalizeSpirvPass()";
  let dependentDialects = ["mlir::spirv::SPIRVDialect"];
}

#endif // __PMLC_TARGET_INTEL_GEN__OCL_SPIRV_PASSES__
