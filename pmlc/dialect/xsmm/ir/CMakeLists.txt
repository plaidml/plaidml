pml_cc_library(
  NAME ir
  HDRS
    dialect.h.inc
    ops.cc.inc
    ops.h.inc
  SRCS
    ops.cc
  DEPS
    ::ops-gen
    pmlc::util
    LLVMSupport
    MLIRAffine
    MLIRIR
  PUBLIC
)

pml_tblgen_library(
  NAME ops-gen
  TD_FILE ops.td
  OUTS
    -gen-enum-decls enums.h.inc
    -gen-enum-defs enums.cc.inc
    -gen-dialect-decls dialect.h.inc
    -gen-op-decls ops.h.inc
    -gen-op-defs ops.cc.inc
)
