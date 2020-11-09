set(OUTPUT_ROOT ${CMAKE_BINARY_DIR})

find_package(InferenceEngineDeveloperPackage)
if(NOT InferenceEngineDeveloperPackage_FOUND)
  set(OPENVINO_SOURCE_DIR "${CMAKE_SOURCE_DIR}/vendor/openvino")
  set(OPENVINO_BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/openvino-build")
  file(MAKE_DIRECTORY ${OPENVINO_BINARY_DIR})

  set(OPENVINO_CMAKE_OPTIONS
    -G ${CMAKE_GENERATOR}
    -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -D ENABLE_CLDNN=OFF
    -D ENABLE_DATA=OFF
    -D ENABLE_FUNCTIONAL_TESTS=ON
    -D ENABLE_GAPI_TESTS=OFF
    -D ENABLE_GNA=OFF
    -D ENABLE_MKL_DNN=OFF
    -D ENABLE_MYRIAD=OFF
    -D ENABLE_PROFILING_ITT=OFF
    -D ENABLE_SPEECH_DEMO=OFF
    -D ENABLE_TESTS=ON
    -D ENABLE_VPU=ON
    -D OUTPUT_ROOT=${OUTPUT_ROOT}
  )

  execute_process(
    COMMAND ${CMAKE_COMMAND} ${OPENVINO_CMAKE_OPTIONS} ${OPENVINO_SOURCE_DIR}
    RESULT_VARIABLE result
    WORKING_DIRECTORY "${OPENVINO_BINARY_DIR}"
  )
  if(result)
    message(FATAL_ERROR "Failed to configure OpenVINO: ${result}")
  endif()

  execute_process(
    COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY "${OPENVINO_BINARY_DIR}"
  )
  if(result)
    message(FATAL_ERROR "Failed to build OpenVINO: ${result}")
  endif()

  set(InferenceEngineDeveloperPackage_DIR ${OPENVINO_BINARY_DIR})
  find_package(InferenceEngineDeveloperPackage REQUIRED)
endif()

file(GLOB_RECURSE _OPS_SRCS CONFIGURE_DEPENDS "ops/*.cpp")

pml_cc_library(
  NAME openvino
  HDRS
    plaidml_ops.hpp
    plaidml_builder.hpp
    plaidml_util.hpp
  SRCS
    plaidml_builder.cpp
    plaidml_util.cpp
    ${_OPS_SRCS}
  INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}
  DEPS
    IE::inference_engine
    IE::ngraph
    plaidml::plaidml
)

add_subdirectory(plugin)
add_subdirectory(tests/functional)
