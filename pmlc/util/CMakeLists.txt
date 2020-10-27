
add_all_subdirs()

pml_mlir_enums(
  NAME
    util
)

file(GLOB _GLOB_UTIL_SRCS ${CMAKE_CURRENT_SOURCE_DIR} *.cc)
file(GLOB _GLOB_UTIL_HDRS ${CMAKE_CURRENT_SOURCE_DIR} *.h)
pml_cc_library(
  NAME
    util
  HDRS
    ${_GLOB_UTIL_HDRS}
  SRCS
    ${_GLOB_UTIL_SRCS}
  DEPS
    pmlc::util::mlir_enums
)