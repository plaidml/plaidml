add_subdirectory(ast)
add_subdirectory(compiler)
add_subdirectory(conversion)
add_subdirectory(dialect)
add_subdirectory(rt)
add_subdirectory(target)
add_subdirectory(testing)
add_subdirectory(tools)
add_subdirectory(transforms)
add_subdirectory(util)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

pml_cc_library(
  NAME init_passes_and_dialects
  HDRS
    all_dialects.h
    all_passes.h
  SRCS
    all_dialects.cc
  DEPS
    ${dialect_libs}
    ${conversion_libs}
    MLIRTransforms
    pmlc::conversion::comp_to_llvm
    pmlc::conversion::gpu
    pmlc::conversion::gpu_to_spirv
    pmlc::conversion::pxa_to_affine
    pmlc::conversion::scf_to_omp
    pmlc::conversion::stdx_to_llvm
    pmlc::conversion::tile_to_pxa
    pmlc::dialect::abi::ir
    pmlc::dialect::affinex::transforms
    pmlc::dialect::comp::ir
    pmlc::dialect::comp::transforms
    pmlc::dialect::layer::ir
    pmlc::dialect::layer::transforms
    pmlc::dialect::pxa::ir
    pmlc::dialect::pxa::transforms
    pmlc::dialect::stdx::ir
    pmlc::dialect::stdx::transforms
    pmlc::dialect::tile::ir
    pmlc::dialect::tile::transforms
    pmlc::dialect::xsmm::ir
    pmlc::rt::all_runtimes
    pmlc::target::all
    pmlc::transforms
)

pml_lit_test(
  NAME lit
  DEPS
    pmlc-jit
    pmlc-opt
  CHECKS
    smoke
    test
)
