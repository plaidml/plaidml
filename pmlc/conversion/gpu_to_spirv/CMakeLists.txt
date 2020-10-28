pml_cc_library(
  NAME gpu_to_spirv
  HDRS
    passes.h
  SRCS
    gpu_to_spirv.cc
  DEPS
    ::passes-gen
    pmlc::dialect::stdx::ir
    pmlc::util
    MLIRGPUToSPIRVTransforms
    MLIRGPU
    MLIRSPIRV
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS
    -gen-pass-decls passes.h.inc
)
