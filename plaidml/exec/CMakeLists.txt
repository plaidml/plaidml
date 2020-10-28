pml_cc_library(
  NAME exec
  HDRS
    exec.h
    ffi.h
  SRCS
    ffi.cc
  DEPS
    plaidml::core
    pmlc::compiler
    pmlc::rt
    pmlc::util
)
