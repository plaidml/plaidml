pml_cc_library(
  NAME vulkan
  HDRS register.h
  SRCS
    vulkan_device.cc
    vulkan_device.h
    vulkan_error.h
    vulkan_invocation.cc
    vulkan_invocation.h
    vulkan_runtime.cc
    vulkan_state.cc
    vulkan_state.h
    wrappers.cc
  DEPS
    pmlc::compiler
    pmlc::rt
    MLIRPass
    MLIRSPIRV
    mlir_runner_utils
    half
    vulkan_headers
    volk
)
