pml_cc_library(
  NAME ir
  HDRS
    attrdef.h.inc
    attrdef.cc.inc
    dialect.h.inc
    dialect.h
  SRCS
    dialect.cc
  DEPS
    MLIRAffine
    MLIRStandard
    pmlc::util
    ::dialect-gen
  PUBLIC
)

pml_tblgen_library(
  NAME dialect-gen
  TD_FILE dialect.td
  OPTS -dialect=pml
  OUTS
    -gen-attrdef-decls attrdef.h.inc
    -gen-attrdef-defs attrdef.cc.inc
    -gen-dialect-decls dialect.h.inc
)
