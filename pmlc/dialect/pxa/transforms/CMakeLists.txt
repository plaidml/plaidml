pmlc_add_mlir_passes(PXA)
add_library(PMLCPXATransforms
  autotile.cc
  dataflow_opt.cc
  fusion.cc
  localize.cc
  pass_detail.h
  resize_tmps.cc
  stencil.cc
  
  test_analysis.cc
  tile.cc
)
#TODO XSMM
#stencil_gemm.cc

target_sources(PMLCPXATransforms
  PUBLIC
    autotile.h
    passes.h
    stencil.h
    tile.h
)

target_link_libraries(PMLCPXATransforms
  PMLCPXAAnalysis
  PMLCEltwise
  PMLCPXA
)
target_compile_options(PMLCPXATransforms PUBLIC ${PMLC_DEFAULT_COPTS})

set_target_properties(PMLCPXATransforms PROPERTIES
  CXX_STANDARD 17
  CXX_EXTENSIONS OFF)
