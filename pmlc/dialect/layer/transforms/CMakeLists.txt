pml_cc_library(
  NAME transforms
  HDRS
    passes.h
  SRCS
    inline_layers.cc
  DEPS
    LLVMSupport
    MLIRIR
    pmlc::dialect::layer::ir
    pmlc::util
    pmlc::util::bilp
    pmlc::util::math
    ::passes-gen
  PUBLIC
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS
    -gen-pass-decls passes.h.inc
)
