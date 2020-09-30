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

def MakeProgram : FunctionPass<"tile-make-program"> {
  let summary = "Forward-only canonicalization pass";
  let constructor = "pmlc::dialect::tile::createMakeProgramPass()";
}

def PadRanges : FunctionPass<"tile-pad-ranges"> {
  let summary = "Pad ranges to make things more even (possibly adding constraints)";
  let constructor = "pmlc::dialect::tile::createPadRangesPass()";
  let options = [
    Option<"minPowerOfTwo", "min-po2", "unsigned", /*default=*/"8",
           "Minimum padding to consider, must be a power of 2">,
    Option<"maxPowerOfTwo", "max-po2", "unsigned", /*default=*/"16",
           "Maximum padding to consider, must be a power of 2">,
    Option<"maxIncrease", "max-increase", "float", /*default=*/"0.05",
           "Maximum allowed increase as a fraction of original size">
  ];
}

def PadConstraints : FunctionPass<"tile-pad-constraints"> {
  let summary = "Pad outputs to remove constraints on inputs";
  let constructor = "pmlc::dialect::tile::createPadConstraintsPass()";
}

#endif // __PMLC_DIALECT_TILE_PASSES__
