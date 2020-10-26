
pml_cc_library(
  NAME
    lib
  SRCS
    "pmlc-opt.cc"
  DATA
    
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

pml_cc_binary(
  NAME
    pmlc-opt
  DEPS
    pmlc::tools::pmlc-opt::lib
)