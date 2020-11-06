 pml_cc_library(
  NAME transforms
  HRDS
    pass_detail.h
    passes.h
  SRCS
    autotile.cc
    buffer_placement.cc
    cache.cc
    cpu_thread.cc
    dataflow_opt.cc
    fusion.cc
    gpu_thread.cc
    layout_utils.cc
    localize.cc
    nest_loops.cc
    normalize.cc
    reorder_layouts.cc
    resize_tmps.cc
    simplify_with_constraints.cc
    stencil_gemm.cc
    stencil.cc
    subgroups.cc
    test_analysis.cc
    tile.cc
    tile_accumulate.cc
    vectorize.cc
  DEPS
    LLVMSupport
    MLIRIR
    pmlc::dialect::pxa::analysis
    pmlc::dialect::pxa::ir
    ::passes-gen
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS -gen-pass-decls passes.h.inc
)
