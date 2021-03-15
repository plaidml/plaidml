pml_cc_binary(
  NAME pmlc-jit
  SRCS pmlc-jit.cc
  DEPS
    MLIRExecutionEngine
    # pmlc::init_passes_and_dialects
    pmlc::rt::all_runtimes
    pmlc::target::all
    pmlc::compiler
    pmlc::rt
    pmlc::rt::llvm
    pmlc::util
)
