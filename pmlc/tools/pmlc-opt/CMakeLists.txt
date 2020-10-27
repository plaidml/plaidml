pml_cc_binary(
  NAME pmlc-opt
  SRCS pmlc-opt.cc
  DEPS
    LLVMSupport
    MLIRIR
    MLIROptLib
    MLIRParser
    MLIRPass
    MLIRTransformUtils
    MLIRSupport
    pmlc::init_passes_and_dialects
    pmlc::util::util
)
