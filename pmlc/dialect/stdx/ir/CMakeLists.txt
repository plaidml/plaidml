file(GLOB GLOB_SRCS ${CMAKE_CURRENT_SOURCE_DIR} *.cc)
pml_mlir_ir(
  NAME
    stdx
  SRCS
    ${GLOB_SRCS}
  DEPS
    MLIROpAsmInterfaceIncGen
)
