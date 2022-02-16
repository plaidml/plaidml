#ifndef __PMLC_CONVERSION_TILE_TO_LINALG_PASSES__
#define __PMLC_CONVERSION_TILE_TO_LINALG_PASSES__

include "mlir/Pass/PassBase.td"

def LowerTileToLinalg : Pass<"convert-tile-to-linalg", "mlir::ModuleOp"> {
  let summary = "Convert Tile dialect to Linalg dialect";
  let constructor = "pmlc::conversion::tile_to_linalg::createLowerTileToLinalgPass()";
  let dependentDialects = [
    "mlir::AffineDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::math::MathDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "pmlc::dialect::layer::LayerDialect",
    "pmlc::dialect::stdx::StdXDialect",
  ];
}

#endif // __PMLC_CONVERSION_TILE_TO_LINALG_PASSES__
