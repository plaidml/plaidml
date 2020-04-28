#ifndef __PMLC_DIALECT_TILE_PASSES__
#define __PMLC_DIALECT_TILE_PASSES__

include "mlir/Pass/PassBase.td"

def ComputeBounds : FunctionPass<"tile-compute-bounds"> {
  let summary = "Compute bounds for contractions";
  let constructor = "pmlc::dialect::tile::createComputeBoundsPass()";
}

def ConstantTypes : FunctionPass<"tile-constant-types"> {
  let summary = "Set constant types precision";
  let constructor = "pmlc::dialect::tile::createConstantTypesPass()";
  let options = [
    Option<"floatKind", "floatx", "std::string",
           /*default=*/"\"f32\"",
           "Set floating-point constant precision (default: f32)">,
    Option<"integerKind", "intx", "std::string",
           /*default=*/"\"si32\"",
           "Set integer constant precision (default: si32)">,
  ];
}

def Pad : FunctionPass<"tile-pad"> {
  let summary = "Pad outputs to remove constraints on inputs";
  let constructor = "pmlc::dialect::tile::createPadPass()";
}

#endif // __PMLC_DIALECT_TILE_PASSES__
