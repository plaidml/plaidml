pml_cc_library(
  NAME gpu
  HDRS
    lowering.h
  SRCS
    kernel_outlining.cc
  DEPS
    ::passes-gen
    pmlc::util
    MLIRGPU
    MLIRGPUToSPIRVTransforms
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS
    -gen-pass-decls passes.h.inc
)
