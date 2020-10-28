pml_cc_library(
  NAME scf_to_omp
  HDRS
    passes.h
  SRCS
    scf_to_omp.cc
  DEPS
    ::passes-gen
    pmlc::util
    MLIRIR
    MLIRSCF
    MLIROpenMP
    MLIRStandard
    MLIRSupport
    MLIRLLVMIRTransforms
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS
    -gen-pass-decls passes.h.inc
)
