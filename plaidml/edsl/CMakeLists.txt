add_subdirectory(tests)

pml_cc_library(
  NAME edsl
  TYPE OBJECT
  HDRS
    edsl.h
    ffi.h
  SRCS
    ffi.cc
  DEPS
    plaidml::core
    pmlc::ast
    pmlc::util
    LLVMSupport
)
