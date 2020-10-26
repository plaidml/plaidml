
file(GLOB _GLOB_UTIL_SRCS ${CMAKE_CURRENT_SOURCE_DIR} *.cc)
file(GLOB _GLOB_UTIL_HDRS ${CMAKE_CURRENT_SOURCE_DIR} *.h)

#PMLCUtilEnumsGen
pmlc_add_mlir_Enums(Util)

pml_cc_library(
  NAME
    env
  HDRS
    "env.h"
  SRCS
    "env.cc"
  PUBLIC
)

pml_cc_library(
  NAME
    logging
  HDRS
    "logging.h"
  SRCS
    "logging.cc"
  #DEPS
  #  "easylogging"
  PUBLIC
)

pml_cc_library(
  NAME
    util
  HDRS
    ${_GLOB_UTIL_HDRS}
  SRCS
    ${_GLOB_UTIL_SRCS}
)