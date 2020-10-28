pml_cc_library(
  NAME edsl
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
