pml_cc_library(
  NAME level_zero
  HDRS register.h
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
    utils.cc
    utils.h
  DEPS
    pmlc::compiler
    pmlc::rt
    MLIRPass
    MLIRSPIRV
    mlir_runner_utils
    LevelZero::LevelZero
)

pml_cc_binary(
    NAME level_zero_probe
    SRCS
      level_zero_probe.cc
    COPTS
      -g
    DEPS
      pmlc::rt::level_zero
      pmlc::rt
)
