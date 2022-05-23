 pml_cc_library(
  NAME transforms
  HRDS
    pass_detail.h
    passes.h
    regulate_depthwise.h
    name_conv_variables.h
  SRCS
    regulate_depthwise.cc
    name_conv_variables.cc
  DEPS
    LLVMSupport
    MLIRIR
    pmlc::dialect::linalgx::ir
    pmlc::util
    ::passes-gen
)

pml_tblgen_library(
  NAME passes-gen
  TD_FILE passes.td
  OUTS -gen-pass-decls passes.h.inc
)
