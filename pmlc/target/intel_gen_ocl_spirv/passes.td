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

def IntelGenOclReorderLayouts : FunctionPass<"intel-gen-ocl-reorder-layouts"> {
  let summary = "Optimize data layouts for Intel GPUs";
  let description = [{
    Specialization of pxa-reorder-layouts for Intel GPUs with OpenCL backend.
    It extends scheduling operation to take into account reversed order
    global and local dimensions are processed by backend.
    Additionally it interleaves global and local loops with sequential ones
    to account for parallel execution - assuming difference of one
    in thread id is equal to one iteration of innermost loop.
    Reorders created by this pass conform to GPU threading semantics with
    separate global and local loops.
  }];
  let constructor = "pmlc::target::intel_gen_ocl_spirv::createIntelGenOclReorderLayoutsPass()";
  let options = [
    Option<"maxThreads", "max-threads", "unsigned", /*default=*/"64",
           "Set maximum/goal inner threads for created reorders">,
    Option<"allowReorder", "allow-reorder", "bool", /*default=*/"false",
           "Allow for creating separate reorder">
  ];
}

#endif // __PMLC_TARGET_INTEL_GEN__OCL_SPIRV_PASSES__
