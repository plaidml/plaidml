 pml_cc_library(
  NAME transforms
  HRDS
    pass_detail.h
    passes.h
  SRCS
    autotile.cc
    dataflow_opt.cc
    fusion.cc
    localize.cc
    resize_tmps.cc
    stencil.cc
    test_analysis.cc
    tile.cc
  DEPS
    LLVMSupport
    MLIRIR
    pmlc::dialect::pxa::ir
    ::passes-gen
)


pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS
    -gen-pass-decls passes.h.inc
)
