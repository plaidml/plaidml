pml_cc_library(
  NAME core
  TYPE OBJECT
  HDRS
    core.h
    ffi.h
  SRCS
    ffi.cc
    settings.cc
    version.cc
  DEPS
    pmlc::ast
    pmlc::compiler
    pmlc::util
    LLVMSupport
)

configure_file(version.in.cc version.cc)
