#ifndef __PMLC_CONVERSION_PXA_TO_AFFINE_PASSES__
#define __PMLC_CONVERSION_PXA_TO_AFFINE_PASSES__

include "mlir/Pass/PassBase.td"

def LowerPXAToAffine : Pass<"convert-pxa-to-affine", "mlir::ModuleOp"> {
  let summary = "Convert PXA dialect to Affine dialect";
  let constructor = "pmlc::conversion::pxa_to_affine::createLowerPXAToAffinePass()";
  let dependentDialects = ["mlir::AffineDialect"];
}

#endif // __PMLC_CONVERSION_PXA_TO_AFFINE_PASSES__
