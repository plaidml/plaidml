pml_cc_library(
  NAME gpu
  HDRS
    passes.h
  SRCS
    kernel_outlining.cc
    gather_gpu_launch_funcs.cc
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
