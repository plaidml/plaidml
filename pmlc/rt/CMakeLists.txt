add_subdirectory(llvm)

pml_cc_library(
  NAME rt
  HDRS
    device_id.h
    executable.h
    runtime.h
    runtime_registry.h
    symbol_registry.h
  SRCS
    boundscheck.cc
    builtin.cc
    device.h
    device_id.cc
    executable.cc
    internal.h
    jit_executable.cc
    jit_executable.h
    prng.cc
    runtime_registry.cc
    symbol_registry.cc
    xsmm.cc
  DEPS
    pmlc::compiler
    pmlc::util
    half
    xsmm
    LLVMExecutionEngine
    LLVMMCJIT
    LLVMOrcJIT
    LLVMSupport
    LLVMX86AsmParser
    LLVMX86CodeGen
    MLIRExecutionEngine
    MLIRTargetLLVMIR
    MLIRTransforms
    mlir_runner_utils
)

set_property(SOURCE xsmm.cc
  PROPERTY COMPILE_DEFINITIONS
    __BLAS=0
    LIBXSMM_DEFAULT_CONFIG
)
