pml_cc_library(
  NAME intel_gen_ocl_spirv
  HDRS
    passes.h
  SRCS
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
    pmlc::conversion::gpu_to_comp
    pmlc::conversion::gpu_to_spirv
    pmlc::conversion::pxa_to_affine
    pmlc::conversion::stdx_to_llvm
    pmlc::conversion::tile_to_pxa
    pmlc::dialect::comp::transforms
    pmlc::dialect::pxa::transforms
    pmlc::dialect::stdx::transforms
    pmlc::dialect::tile::transforms
    pmlc::target::x86::prng_lowering
    MLIRAffineToStandard
    MLIRAffineTransforms
    MLIRGPU
    MLIRGPUToSPIRVTransforms
    MLIRLLVMIRTransforms
    MLIRSCFToGPU
    MLIRStandardOpsTransforms
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS
    -gen-pass-decls passes.h.inc
)
