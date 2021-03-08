pml_cc_library(
  NAME pxa_to_affine
  HDRS
    passes.h
  SRCS
    pxa_to_affine.cc
  DEPS
    ::passes-gen
    pmlc::dialect::pxa::ir
    pmlc::dialect::tile::ir
    pmlc::dialect::stdx::ir
    pmlc::util
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS
    -gen-pass-decls passes.h.inc
)
