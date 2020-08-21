#ifndef __PMLC_CONVERSION_TILE_TO_PXA_PASSES__
#define __PMLC_CONVERSION_TILE_TO_PXA_PASSES__

include "mlir/Pass/PassBase.td"

def LowerTileToPXA : Pass<"convert-tile-to-pxa", "mlir::ModuleOp"> {
  let summary = "Convert Tile dialect to PXA dialect";
  let constructor = "pmlc::conversion::tile_to_pxa::createLowerTileToPXAPass()";
  let dependentDialects = [
    "dialect::pxa::PXADialect",
    "dialect::stdx::StdXDialect",
    "mlir::AffineDialect"
  ];
}

#endif // __PMLC_CONVERSION_TILE_TO_PXA_PASSES__
