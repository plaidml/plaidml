add_subdirectory(tests)
add_subdirectory(examples)

set(_SRCS ffi.cc)
set(_HDRS
  edsl.h
  ffi.h
)

pml_cc_library(
  NAME edsl
  TYPE STATIC
  HDRS ${_HDRS}
  SRCS ${_SRCS}
  DEPS
    plaidml::core
    pmlc::ast
    pmlc::util
    pmlc::dialect::tile::ir
    pmlc::dialect::tile::transforms
    LLVMSupport
  COMPONENTS
    devkit
)

pml_cc_library(
  NAME obj
  TYPE OBJECT
  HDRS ${_HDRS}
  SRCS ${_SRCS}
  DEPS
    plaidml::core::obj
    pmlc::ast
    pmlc::util
    pmlc::dialect::tile::ir
    pmlc::dialect::tile::transforms
    LLVMSupport
)

pml_py_library(
  NAME py
  SRCS
    __init__.py
)
