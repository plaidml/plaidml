pml_cc_library(
  NAME tile_to_linalg
  HDRS
    passes.h
  SRCS
    tile_to_linalg.cc
    utils.cc
  DEPS
    ::passes-gen
    pmlc::dialect::layer::ir
    pmlc::dialect::pxa::analysis
    pmlc::dialect::stdx::ir
    pmlc::dialect::tile::ir
    pmlc::util
    MLIRIR
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS
    -gen-pass-decls passes.h.inc
)
