pml_cc_library(
  NAME transforms
  HDRS
    pass_detail.h
    passes.h
  SRCS
    boundscheck.cc
    i1_to_i32.cc
  DEPS
    LLVMSupport
    MLIRIR
    ::passes-gen
    pmlc::dialect::stdx::ir
    pmlc::util::bilp
    pmlc::util::math
  PUBLIC
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS
    -gen-pass-decls passes.h.inc
)
