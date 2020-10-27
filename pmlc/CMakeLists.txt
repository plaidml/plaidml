add_all_subdirs()

pml_cc_library(
  NAME
    all_passes_and_dialects
  DEPS
    pmlc::conversion::all_conversions
    pmlc::dialect::all_dialects
    
    MLIRAffinePassIncGen
    MLIRConversionPassIncGen
    MLIRGPUPassIncGen
    MLIRLinalgPassIncGen
    MLIRLLVMPassIncGen
    MLIRTransformsPassIncGen
    MLIRStandardTransformsIncGen
)

pml_cc_library(
  NAME
    pmlc
  DEPS
    ::all_passes_and_dialects
    pmlc::targets::all_
)