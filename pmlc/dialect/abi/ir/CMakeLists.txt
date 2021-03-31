pml_cc_library(
  NAME ir
  HDRS
    dialect.h.inc
    ops.cc.inc
    ops.h.inc
  SRCS
    dialect.cc
    loop_op.cc
  DEPS
    ::ops-gen
  PUBLIC
)

pml_tblgen_library(
  NAME ops-gen
  TD_FILE ops.td
  OPTS -dialect=abi
  OUTS
    -gen-op-decls ops.h.inc
    -gen-op-defs ops.cc.inc
    -gen-dialect-decls dialect.h.inc
)
