add_subdirectory(utils)

pml_cc_library(
  NAME level_zero
  SRCS
    common_wrappers.cc
    level_zero_device.cc
    level_zero_device.h
    level_zero_invocation.cc
    level_zero_invocation.h
    level_zero_runtime.cc
    level_zero_wrappers.cc
    level_zero_utils.h
    level_zero_utils.cc
  DEPS
    pmlc::compiler
    pmlc::rt
    MLIRPass
    MLIRSPIRV
    mlir_runner_utils
    lz_wrapper
    LevelZero::LevelZero
)
