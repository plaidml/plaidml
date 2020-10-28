add_subdirectory(lib)

pml_cc_library(
  NAME op
  HDRS
    ffi.h
    op.h
  SRCS
    ffi.cc
  DEPS
    plaidml::core
    plaidml::edsl
    plaidml::op::lib
    pmlc::util
    LLVMSupport
)
