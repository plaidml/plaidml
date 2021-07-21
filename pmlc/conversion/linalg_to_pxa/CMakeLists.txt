pml_cc_library(
  NAME linalg_to_pxa
  HDRS
    passes.h
  SRCS
    specials.cc
    linalg_to_pxa.cc
    pooling.cc
    reshape.cc
    utils.cc
  DEPS
    ::passes-gen
    pmlc::dialect::layer::ir
    pmlc::dialect::pxa::analysis
    pmlc::dialect::pxa::ir
    pmlc::dialect::stdx::ir
    pmlc::util
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS
    -gen-pass-decls passes.h.inc
)
