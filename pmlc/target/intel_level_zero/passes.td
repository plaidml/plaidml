// Copyright 2021, Intel Corporation
#ifndef __PMLC_TARGET_INTEL_LEVEL_ZERO_PASSES__
#define __PMLC_TARGET_INTEL_LEVEL_ZERO_PASSES__

include "mlir/Pass/PassBase.td"

def IntelLevelZeroAddSpirvTarget : Pass<"intel-level-zero-add-spirv-target", "mlir::ModuleOp"> {
  let summary = "Add spirv target environment attributes";
  let constructor = "pmlc::target::intel_level_zero::createAddSpirvTargetPass()";
  let dependentDialects = ["mlir::spirv::SPIRVDialect"];
  let options = [
    Option<"spirvVersion", "spirv-version", "unsigned", /*default=*/"150",
           "SPIR-V Version">,
  ];
}

#endif // __PMLC_TARGET_INTEL_LEVEL_ZERO_PASSES__
