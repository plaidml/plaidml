pml_cc_library(
  NAME gpu_to_comp
  HDRS
    passes.h
  SRCS
    gpu_to_comp.cc
  DEPS
    ::passes-gen
    pmlc::dialect::comp::ir
    pmlc::util
    MLIRGPU
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRSupport
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS
    -gen-pass-decls passes.h.inc
)
