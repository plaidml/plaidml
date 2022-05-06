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
    #MLIRBufferizationTransforms
    #MLIRBufferization
    # This is from mlir-hlo. Should be included here?
    # It does not link without.
    MLIRBufferTransforms
    MLIRSupport
    pmlc::init_passes_and_dialects
    pmlc::util
)
