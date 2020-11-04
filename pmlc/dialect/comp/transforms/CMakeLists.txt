pml_cc_library(
  NAME transforms
  HRDS
    pass_detail.h
    passes.h
  SRCS
    execenv_coalescing.cc
    minimize_allocations.cc
    recalculate_event_deps.cc
    remove_redundant_rw.cc
  DEPS
    LLVMSupport
    MLIRIR
    pmlc::dialect::comp::analysis
    pmlc::dialect::comp::ir
    ::passes-gen
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS -gen-pass-decls passes.h.inc
)
