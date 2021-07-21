pml_cc_library(
  NAME transforms
  HDRS
    passes.h
  SRCS
    apply_rules.cc
    load_module.cc
  DEPS
    LLVMSupport
    MLIRIR
    pmlc::dialect::pml::ir
    pmlc::util
    ::passes-gen
  PUBLIC
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS
    -gen-pass-decls passes.h.inc
)
