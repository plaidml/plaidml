pml_cc_library(
  NAME ir
  HDRS
    dialect.h.inc
    interfaces.cc.inc
    interfaces.h.inc
    ops.cc.inc
    ops.h.inc
    ops.h
    types.h
    util.h
  SRCS
    dialect.cc
    eltwise.cc
    ops.cc
    util.cc
  DEPS
    pmlc::util
    ::interfaces-gen
    ::ops-gen
  PUBLIC
)

pml_tblgen_library(
  NAME ops-gen
  TD_FILE ops.td
  OPTS -dialect=tile
  OUTS
    -gen-op-decls ops.h.inc
    -gen-op-defs ops.cc.inc
    -gen-dialect-decls dialect.h.inc
)

pml_tblgen_library(
  NAME interfaces-gen
  TD_FILE interfaces.td
  OUTS
    -gen-op-interface-decls interfaces.h.inc
    -gen-op-interface-defs interfaces.cc.inc
)
