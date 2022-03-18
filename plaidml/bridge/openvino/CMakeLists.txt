set(OUTPUT_ROOT ${CMAKE_BINARY_DIR})
set(BUILD_SHARED_LIBS ON)

if(NOT InferenceEngineDeveloperPackage_FOUND)
  find_package(InferenceEngineDeveloperPackage REQUIRED)
endif()

file(GLOB_RECURSE _OPS_SRCS CONFIGURE_DEPENDS "ops/*.cpp")

pml_cc_library(
  NAME openvino
  HDRS
    plaidml_builder.hpp
    plaidml_ops.hpp
    plaidml_util.hpp
  SRCS
    plaidml_builder.cpp
    plaidml_ops.cpp
    plaidml_util.cpp
    ${_OPS_SRCS}
  INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}
  DEFINES
    HALF_ENABLE_CPP11_TYPE_TRAITS
  DEPS
    IE::inference_engine
    IE::ngraph
    IE::runtime
    half
)

add_subdirectory(plugin)
#if(PML_BUILD_TESTS)
#  add_subdirectory(tests/functional)
#endif()

install(
  DIRECTORY ${OpenVINO_MAIN_SOURCE_DIR}/${BIN_FOLDER}/
  DESTINATION "testkit"
  COMPONENT testkit
  USE_SOURCE_PERMISSIONS
  FILES_MATCHING
    PATTERN "benchmark_app"
)

install(
  DIRECTORY ${OpenVINO_MAIN_SOURCE_DIR}/${BIN_FOLDER}/lib/
  DESTINATION "testkit"
  COMPONENT testkit
  USE_SOURCE_PERMISSIONS
  FILES_MATCHING
    PATTERN "libformat_reader.so"
)

install(
  DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/
  DESTINATION "testkit"
  COMPONENT testkit
  USE_SOURCE_PERMISSIONS
  FILES_MATCHING
    PATTERN "libplaidml*"
    PATTERN "plugins.xml"
)

install(
  DIRECTORY ${CMAKE_BINARY_DIR}/${BIN_FOLDER}/lib/
  DESTINATION "testkit"
  COMPONENT testkit
  USE_SOURCE_PERMISSIONS
  FILES_MATCHING
    PATTERN "*.so"
)

install(
    DIRECTORY ${OpenCV_INSTALL_PATH}/lib/
    DESTINATION "testkit"
    COMPONENT testkit
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING
      PATTERN "libopencv_core.so*"
      PATTERN "libopencv_img*.so*"
)

install(
    DIRECTORY ${_tbb_root}/lib/
    DESTINATION "testkit"
    COMPONENT testkit
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING
      PATTERN "libtbb.so*"
)

add_custom_target(benchmark_app ALL)

add_custom_command(
  TARGET benchmark_app
  COMMAND ${CMAKE_COMMAND} -E copy
    ${OpenVINO_MAIN_SOURCE_DIR}/${BIN_FOLDER}/benchmark_app
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmark_app
)
