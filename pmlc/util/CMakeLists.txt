add_subdirectory(bilp)
add_subdirectory(math)

pml_cc_library(
  NAME util
  HDRS
    buffer.h
    enums.h
    enums.h.inc
    enums.cc.inc
    env.h
    ident.h
    logging.h
    matchers.h
    shape.h
    strides.h
    tags.h
    util.h
  SRCS
    enums.cc
    env.cc
    ident.cc
    logging.cc
    shape.cc
    strides.cc
    tags.cc
    util.cc
  DEPS
    easyloggingpp
    MLIRIR
    MLIRStandard
    ::enums-gen
  PUBLIC
)

pml_tblgen_library(
  NAME enums-gen
  TD_FILE enums.td
  OUTS
    -gen-enum-decls enums.h.inc
    -gen-enum-defs enums.cc.inc
)
