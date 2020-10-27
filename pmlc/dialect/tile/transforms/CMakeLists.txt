file(GLOB GLOB_SRCS ${CMAKE_CURRENT_SOURCE_DIR} *.cc)
pml_mlir_transforms(
  SRCS
    ${GLOB_SRCS}
  DEPS
    pmlc::dialect::tile::ir
)
