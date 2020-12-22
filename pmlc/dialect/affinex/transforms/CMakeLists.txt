pml_cc_library(
  NAME transforms
  HDRS
    pass_detail.h
    passes.h
  SRCS
    dead_memref_elimination.cc
    loop_unroll.cc
    memref_dataflow_opt.cc
  DEPS
    ::passes-gen
  PUBLIC
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS
    -gen-pass-decls passes.h.inc
)
