pml_cc_library(
  NAME stdx_to_llvm
  HDRS
    passes.h
  SRCS
    stdx_to_llvm.cc
  DEPS
    ::passes-gen
    pmlc::dialect::stdx::ir
    pmlc::util
    MLIRLLVMIRTransforms
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS
    -gen-pass-decls passes.h.inc
)
