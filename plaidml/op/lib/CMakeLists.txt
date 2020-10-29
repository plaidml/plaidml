pml_cc_library(
  NAME lib
  TYPE OBJECT
  HDRS ops.h
  SRCS ops.cc
  DEPS
    plaidml::core
    plaidml::edsl
    pmlc::util
    LLVMSupport
)
