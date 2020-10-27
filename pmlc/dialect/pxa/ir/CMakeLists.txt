file(GLOB GLOB_SRCS ${CMAKE_CURRENT_SOURCE_DIR} *.cc)
pml_mlir_ir(
  NAME
    pxa
  SRCS
    ${GLOB_SRCS}
  DEPS
    pmlc::util
)