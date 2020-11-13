pml_cc_library(
  NAME transforms
  HRDS
    pass_detail.h
    passes.h
  SRCS
    lower_to_abi.cc
  DEPS
    MLIRIR
    pmlc::dialect::abi::ir
    pmlc::util
    ::passes-gen
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS -gen-pass-decls passes.h.inc
)
