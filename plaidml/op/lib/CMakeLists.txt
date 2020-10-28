pml_cc_library(
  NAME lib
  HDRS ops.h
  SRCS ops.cc
  DEPS
    plaidml::core
    plaidml::edsl
    pmlc::util
    LLVMSupport
)
