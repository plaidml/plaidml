
pml_cc_binary(
  NAME
    pmlc-opt
  SRCS
    "pmlc-opt.cc"
  DEPS
    MLIRIR
    MLIRAffine
    MLIRAffineToStandard
    MLIRLinalg
    MLIRParser
    MLIRPass
    MLIRConversion
    MLIRStandard
    MLIRSupport
    MLIRTransforms
    
    pmlc::all_passes_and_dialects
    pmlc::util::util
)