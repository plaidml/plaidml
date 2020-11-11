set(OUTPUT_ROOT ${CMAKE_BINARY_DIR})

find_package(InferenceEngineDeveloperPackage)
if(NOT InferenceEngineDeveloperPackage_FOUND)
  CPMAddPackage(
    NAME openvino
    GITHUB_REPOSITORY plaidml/openvino
    GIT_TAG 89d83c2858e346dc2e3c0c1d53376fa5ddb8ea26
    DOWNLOAD_ONLY ON
  )

  set(CI_BUILD_NUMBER "0")
  set(ENABLE_CLDNN OFF)
  set(ENABLE_CPPLINT OFF)
  set(ENABLE_DATA OFF)
  set(ENABLE_FUNCTIONAL_TESTS ON)
  set(ENABLE_GAPI_TESTS OFF)
  set(ENABLE_GNA OFF)
  set(ENABLE_MKL_DNN OFF)
  set(ENABLE_MYRIAD OFF)
  set(ENABLE_PROFILING_ITT OFF)
  set(ENABLE_SPEECH_DEMO OFF)
  set(ENABLE_TESTS ON)
  set(ENABLE_VPU ON)
  set(TREAT_WARNING_AS_ERROR OFF)

  add_subdirectory(${openvino_SOURCE_DIR} ${openvino_BINARY_DIR} EXCLUDE_FROM_ALL)

  set(BIN_FOLDER bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
  set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
  set(CMAKE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})

  set(OpenVINO_MAIN_SOURCE_DIR ${openvino_SOURCE_DIR})
  set(IE_MAIN_SOURCE_DIR ${OpenVINO_MAIN_SOURCE_DIR}/inference-engine)
  add_definitions(-DIE_BUILD_POSTFIX=\"${IE_BUILD_POSTFIX}\")
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
