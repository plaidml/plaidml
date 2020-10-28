pml_cc_library(
  NAME core
  HDRS
    core.h
    ffi.h
  SRCS
    ffi.cc
    settings.cc
  DEPS
    pmlc::ast
    pmlc::compiler
    pmlc::util
    LLVMSupport
)
