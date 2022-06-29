 pml_cc_library(
  NAME transforms
  HRDS
    pass_detail.h
    passes.h
  SRCS
    autotile.cc
    alloca_conversion.cc
    cache.cc
    convert_mem_op.cc
    cpu_thread.cc
    dataflow_opt.cc
    dealloc_placement.cc
    fusion.cc
    gpu_thread.cc
    layout_utils.cc
    localize.cc
    nest_loops.cc
    normalize.cc
    resize_tmps.cc
    simplify_arithmetic.cc
    stencil_gemm.cc
    stencil.cc
    subgroups.cc
    test_analysis.cc
    tile.cc
    tile_accumulate.cc
    vectorize.cc
    vectorize_mem.cc
  DEPS
    LLVMSupport
    MLIRAffineTransforms
    MLIRIR
    pmlc::dialect::pml::ir
    pmlc::dialect::pxa::analysis
    pmlc::dialect::pxa::ir
    pmlc::util
    ::passes-gen
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS -gen-pass-decls passes.h.inc
)
