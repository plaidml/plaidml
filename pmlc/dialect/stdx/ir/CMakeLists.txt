pml_cc_library(
  NAME ir
  HDRS
    dialect.h.inc
    ops.cc.inc
    ops.h.inc
  SRCS
    ops.cc
  DEPS
    pmlc::util
    ::ops-gen
  PUBLIC
)

pml_tblgen_library(
  NAME ops-gen
  TD_FILE ops.td
  OPTS -dialect=stdx
  OUTS
    -gen-op-decls ops.h.inc
    -gen-op-defs ops.cc.inc
    -gen-dialect-decls dialect.h.inc
)
