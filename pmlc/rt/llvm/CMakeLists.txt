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
    # ::openmp
    omp
)

# pml_cc_library(
#   NAME openmp
#   SRCS
#     openmp.cc
#   DEPS
#     pmlc::rt
#     omp
# )
