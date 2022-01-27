pml_cc_library(
  NAME tile_to_pxa
  SRCS
    specials.cc
    utils.cc
  DEPS
    pmlc::dialect::layer::ir
    pmlc::dialect::pxa::analysis
    pmlc::dialect::pxa::ir
    pmlc::dialect::stdx::ir
    pmlc::dialect::tile::ir
    pmlc::util
)
