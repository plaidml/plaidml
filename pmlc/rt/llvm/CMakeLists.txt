pml_cc_library(
  NAME llvm
  SRCS
    builtin.cc
    device.cc
    runtime.cc
  DEPS
    pmlc::compiler
    pmlc::rt
    # ::openmp
)

# pml_cc_library(
#   NAME openmp
#   SRCS
#     openmp.cc
#   DEPS
#     pmlc::rt
#     # LLVMOpenMP
# )
