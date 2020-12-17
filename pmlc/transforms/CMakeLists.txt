pml_cc_library(
  NAME transforms
  HRDS
    pass_detail.h
    passes.h
  SRCS
    hoisting.cc
  DEPS
    LLVMSupport
    MLIRIR
    pmlc::util
    pmlc::dialect::stdx::ir
    pmlc::dialect::tile::ir
    ::passes-gen
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS -gen-pass-decls passes.h.inc
)
