#file(GLOB GLOB_SRCS ${CMAKE_CURRENT_SOURCE_DIR} *.cc)
LIST(APPEND PXA_TXF_SRCS
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

#TODO - stencil / xsmm

pml_mlir_transforms(
  SRCS
    ${PXA_TXF_SRCS}
  DEPS
    pmlc::dialect::pxa::ir
)
