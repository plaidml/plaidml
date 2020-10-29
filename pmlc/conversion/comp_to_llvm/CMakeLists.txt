pml_cc_library(
  NAME comp_to_llvm
  HDRS
    passes.h
  SRCS
    common.cc
    comp_to_ocl.cc
    comp_to_vulkan.cc
    utils.cc
  DEPS
    ::passes-gen
    pmlc::dialect::comp::ir
    pmlc::util
    MLIRLLVMIR
    MLIRSPIRV
    MLIRStandard
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRSupport
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS
    -gen-pass-decls passes.h.inc
)
