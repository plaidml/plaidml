pml_cc_library(
  NAME level_zero
  HDRS register.h
  SRCS
    level_zero_device.cc
    level_zero_device.h
    level_zero_invocation.cc
    level_zero_invocation.h
    level_zero_runtime.cc
    level_zero_wrappers.cc
    level_zero_utils.h
    level_zero_utils.cc
  DEFINES
    PML_USE_DEVICE_LEVEL_ZERO
  DEPS
    pmlc::compiler
    pmlc::rt
    MLIRPass
    MLIRSPIRV
    mlir_runner_utils
    ze_loader
)

pml_cc_binary(
    NAME level_zero_probe
    SRCS
      level_zero_probe.cc
    DEPS
      pmlc::rt::level_zero
      pmlc::rt
)
