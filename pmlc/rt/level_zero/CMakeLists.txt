add_subdirectory(utils)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils/include)

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
    pmlc::rt::level_zero::utils::lz_wrapper
    LevelZero::LevelZero
)
