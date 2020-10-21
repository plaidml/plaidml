#ifndef __PMLC_CONVERSION_SCF_TO_OMP_PASSES__
#define __PMLC_CONVERSION_SCF_TO_OMP_PASSES__

include "mlir/Pass/PassBase.td"

def LowerSCFToOpenMP : Pass<"convert-scf-to-omp", "mlir::ModuleOp"> {
  let summary = "Convert SCF parallel loop to the OpenMP dialect";
  let constructor = "pmlc::conversion::scf_to_omp::createLowerSCFToOpenMPPass()";
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::omp::OpenMPDialect"
  ];
}

#endif // __PMLC_CONVERSION_SCF_TO_OMP_PASSES__
