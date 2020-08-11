add_library(PMLCPXAAnalysis 
  affine_expr.cc
  strides.cc
  uses.cc
  )

target_sources(PMLCPXAAnalysis
  PUBLIC
    affine_expr.h
    strides.h
    uses.h
)

target_link_libraries(PMLCPXAAnalysis
  PMLCPXA
)
target_compile_options(PMLCPXAAnalysis PUBLIC ${PMLC_DEFAULT_COPTS})

set_target_properties(PMLCPXAAnalysis PROPERTIES
  CXX_STANDARD 17
  CXX_EXTENSIONS OFF)
