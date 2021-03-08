pml_cc_library(
  NAME intel_gen_ocl_spirv
  HDRS
    passes.h
    pipeline.h
  SRCS
    access_qualifiers.cc
    legalize_spirv.cc
    pipeline.cc
    reorder_layouts.cc
    spirv_target.cc
    sub_group_size.cc
  DEPS
    ::passes-gen
    pmlc::compiler
    pmlc::conversion::comp_to_llvm
    pmlc::conversion::gpu
    pmlc::conversion::gpu_to_spirv
    pmlc::conversion::pxa_to_affine
    pmlc::conversion::stdx_to_llvm
    pmlc::conversion::tile_to_pxa
    pmlc::dialect::affinex::transforms
    pmlc::dialect::comp::transforms
    pmlc::dialect::layer::transforms
    pmlc::dialect::pxa::transforms
    pmlc::dialect::stdx::transforms
    pmlc::dialect::tile::transforms
    pmlc::transforms
    pmlc::target::intel_gen
    pmlc::target::x86::prng_lowering
    MLIRAffineToStandard
    MLIRAffineTransforms
    MLIRGPU
    MLIRGPUToSPIRV
    MLIRLLVMIRTransforms
    MLIRSCFToGPU
    MLIRStandardOpsTransforms
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS -gen-pass-decls passes.h.inc
)
