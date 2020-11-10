set(OUTPUT_ROOT ${CMAKE_BINARY_DIR})

function(git_command CWD)
  string(JOIN " " _COMMAND ${ARGN})
  message("git ${_COMMAND}")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} ${ARGN}
    RESULT_VARIABLE result
    WORKING_DIRECTORY "${CWD}"
    OUTPUT_QUIET
  )
  if(result)
    message(FATAL_ERROR "git command failed: \"${_COMMAND}\"")
  endif()
endfunction()

find_package(InferenceEngineDeveloperPackage)
if(NOT InferenceEngineDeveloperPackage_FOUND)
  set(OPENVINO_GIT_REMOTE git@github.com:plaidml/openvino.git)
  set(OPENVINO_GIT_TAG    a2d0a79535e96ba7ae6f552cbda670501d6f2cad)
  set(OPENVINO_SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/openvino-src")
  set(OPENVINO_BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/openvino-build")
  file(MAKE_DIRECTORY ${OPENVINO_SOURCE_DIR})
  file(MAKE_DIRECTORY ${OPENVINO_BINARY_DIR})

  set(OPENVINO_CMAKE_OPTIONS
    -G ${CMAKE_GENERATOR}
    -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -D CMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
    -D CMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
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
    -D TREAT_WARNING_AS_ERROR=OFF
  )

  if(NOT IS_DIRECTORY "${OPENVINO_SOURCE_DIR}/.git")
    git_command(${OPENVINO_SOURCE_DIR} init)
    git_command(${OPENVINO_SOURCE_DIR} remote add origin ${OPENVINO_GIT_REMOTE})
  endif()

  git_command(${OPENVINO_SOURCE_DIR} fetch --depth=1 --prune --tags origin ${OPENVINO_GIT_TAG})
  git_command(${OPENVINO_SOURCE_DIR} checkout --force ${OPENVINO_GIT_TAG})
  git_command(${OPENVINO_SOURCE_DIR} submodule update --init --depth=1 --recursive --force)

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
