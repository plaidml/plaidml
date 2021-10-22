pml_cc_library(
  NAME compiler
  HDRS
    program.h
    registry.h
  SRCS
    program.cc
    registry.cc
  DEPS
    pmlc::util
    pmlc::init_passes_and_dialects
    MLIRParser
)
