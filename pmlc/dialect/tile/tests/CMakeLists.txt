file(GLOB _GLOB_X_MLIR LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS *.mlir)

pml_lit_test_suite(
  SRCS ${_GLOB_X_MLIR}
  DATA
    pmlc::tools::FileCheck
    pmlc::tools::pmlc-opt
)
