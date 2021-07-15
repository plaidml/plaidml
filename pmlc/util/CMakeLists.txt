add_subdirectory(bilp)
add_subdirectory(math)

pml_cc_library(
  NAME util
  HDRS
    buffer.h
    dialect.h.inc
    enums.h
    enums.h.inc
    enums.cc.inc
    env.h
    ident.h
    logging.h
    matchers.h
    memuse.h
    shape.h
    schedule.h
    schedule_attrdef.h.inc
    schedule_attrdef.cc.inc
    strides.h
    tags.h
    util.h
  SRCS
    enums.cc
    env.cc
    ident.cc
    logging.cc
    memuse.cc
    schedule.cc
    shape.cc
    strides.cc
    tags.cc
    util.cc
  DEPS
    easyloggingpp
    LLVMCore
    MLIRAffine
    MLIRIR
    MLIRStandard
    ::enums-gen
    ::schedule-gen
  PUBLIC
)

pml_tblgen_library(
  NAME enums-gen
  TD_FILE enums.td
  OUTS
    -gen-enum-decls enums.h.inc
    -gen-enum-defs enums.cc.inc
)

pml_tblgen_library(
  NAME schedule-gen
  TD_FILE schedule.td
  OPTS -dialect=pml
  OUTS
    -gen-attrdef-decls schedule_attrdef.h.inc
    -gen-attrdef-defs schedule_attrdef.cc.inc
    -gen-dialect-decls dialect.h.inc
)
