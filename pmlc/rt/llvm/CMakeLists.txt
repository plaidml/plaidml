pml_cc_library(
  NAME llvm
  SRCS
    builtin.cc
    device.cc
    openmp.cc
    runtime.cc
  DEPS
    pmlc::compiler
    pmlc::rt
    omp
)
