#ifndef __PMLC_CONVERSION_AFFINE_TO_OMP_PASSES__
#define __PMLC_CONVERSION_AFFINE_TO_OMP_PASSES__

include "mlir/Pass/PassBase.td"

def LowerAffineToOpenMP : Pass<"convert-affine-to-omp", "mlir::ModuleOp"> {
  let summary = "Convert affine to the OpenMP dialect";
  let constructor = "pmlc::conversion::affine_to_omp::createLowerAffineToOpenMPPass()";
  let dependentDialects = [
    "mlir::AffineDialect",
    "mlir::omp::OpenMPDialect"
  ];
}

#endif // __PMLC_CONVERSION_AFFINE_TO_OMP_PASSES__
