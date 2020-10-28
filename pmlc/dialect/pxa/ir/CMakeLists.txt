pml_cc_library(
  NAME ir
  HDRS
    dialect.h.inc
    interfaces.h.inc
    matchers.h
    ops.cc.inc
    ops.h.inc
  SRCS
    interfaces.cc
    ops.cc
  DEPS
    pmlc::util
    ::interfaces-gen
    ::ops-gen
  PUBLIC
)

pml_tblgen_library(
  NAME interfaces-gen
  TD_FILE interfaces.td
  OUTS
    -gen-op-interface-decls interfaces.h.inc
    -gen-op-interface-defs interfaces.cc.inc
)

pml_tblgen_library(
  NAME ops-gen
  TD_FILE ops.td
  OPTS -dialect=pxa
  OUTS
    -gen-op-decls ops.h.inc
    -gen-op-defs ops.cc.inc
    -gen-dialect-decls dialect.h.inc
)
