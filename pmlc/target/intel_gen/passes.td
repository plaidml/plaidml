#ifndef __PMLC_TARGET_INTEL_GEN_PASSES__
#define __PMLC_TARGET_INTEL_GEN_PASSES__

include "mlir/Pass/PassBase.td"

def IntelGenLowerAffine : FunctionPass<"intel-gen-lower-affine"> {
  let summary = "Lower affine -> std + scf, and add GPU mappings ";
  let constructor = "pmlc::target::x86::createIntelGenLowerAffinePass()";
  let dependentDialects = [
    "mlir::scf::SCFDialect"
  ];
}

#endif

