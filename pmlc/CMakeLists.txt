add_subdirectory(dialect)
add_subdirectory(tools)
add_subdirectory(util)

pml_cc_library(
  NAME init_passes_and_dialects
  HDRS
    all_dialects.h
    all_passes.h
  SRCS
    all_dialects.cc
  DEPS
    # pmlc::conversion::all_conversions
    # pmlc::dialect::all_dialects
    MLIRAffine
    MLIRAffineTransforms
    MLIRGPU
    MLIRGPUToSPIRVTransforms
    MLIRLinalg
    MLIRLinalgTransforms
    MLIRStandardOps
    MLIRStandardOpsTransforms
    MLIRTransforms
    MLIRVector
)
