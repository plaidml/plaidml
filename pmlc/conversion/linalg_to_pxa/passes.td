#ifndef __PMLC_CONVERSION_LINALG_TO_PXA_PASSES__
#define __PMLC_CONVERSION_LINALG_TO_PXA_PASSES__

include "mlir/Pass/PassBase.td"

def LowerLinalgToPXA : Pass<"convert-linalg-to-pxa", "mlir::ModuleOp"> {
  let summary = "Convert Linalg dialect to PXA dialect";
  let constructor = "pmlc::conversion::linalg_to_pxa::createLowerLinalgToPXAPass()";
  let dependentDialects = [
    "mlir::AffineDialect",
    "mlir::math::MathDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "pmlc::dialect::layer::LayerDialect",
    "pmlc::dialect::pxa::PXADialect",
    "pmlc::dialect::stdx::StdXDialect",
  ];
}

#endif // __PMLC_CONVERSION_LINALG_TO_PXA_PASSES__
