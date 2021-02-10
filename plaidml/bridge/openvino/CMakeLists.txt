set(OUTPUT_ROOT ${CMAKE_BINARY_DIR})

# find_package(InferenceEngineDeveloperPackage) does not support being called
# multiple times.
if(NOT InferenceEngineDeveloperPackage_FOUND)
  find_package(InferenceEngineDeveloperPackage QUIET)
endif()

if(NOT InferenceEngineDeveloperPackage_FOUND)
  set(BUILD_SHARED_LIBS ON)
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
  set(NGRAPH_ONNX_IMPORT_ENABLE OFF)
  set(TREAT_WARNING_AS_ERROR OFF)

  message("Fetching openvino")
  FetchContent_Declare(
    openvino
    GIT_REPOSITORY https://github.com/plaidml/openvino.git
    GIT_TAG        c03f8d8b2b5f87f116a8032683f8a6e559628404
  )
  if(LOCAL_OPENVINO_DIR)
    set(openvino_SOURCE_DIR ${LOCAL_OPENVINO_DIR})
    set(openvino_BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/openvino-build)
  elseif(NOT openvino_POPULATED)
    FetchContent_Populate(openvino)
  endif()
  add_subdirectory(${openvino_SOURCE_DIR} ${openvino_BINARY_DIR} EXCLUDE_FROM_ALL)

  set(BIN_FOLDER "bin/intel64")
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
    plaidml_builder.hpp
    plaidml_ops.hpp
    plaidml_util.hpp
  SRCS
    plaidml_builder.cpp
    plaidml_ops.cpp
    plaidml_util.cpp
    ${_OPS_SRCS}
  INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}
  DEPS
    IE::inference_engine
    IE::ngraph
)

if(NOT InferenceEngineDeveloperPackage_FOUND)
  add_subdirectory(benchmark_app)
endif()
add_subdirectory(plugin)
if(PML_BUILD_TESTS)
  add_subdirectory(tests/functional)
endif()

ie_cpack_add_component(BENCHMARK_APP REQUIRED)
install(
    PROGRAMS
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmark_app
    DESTINATION bin
    COMPONENT BENCHMARK_APP
)

install(
    DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/
    DESTINATION bin/lib
    COMPONENT BENCHMARK_APP
    USE_SOURCE_PERMISSIONS
    PATTERN "*.a" EXCLUDE
)

install(
    PROGRAMS
        ${OUTPUT_ROOT}/bin/libplaidml.so
        ${OUTPUT_ROOT}/bin/libplaidml.so.1
        ${OpenVINO_MAIN_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib/libformat_reader.so
        ${IE_MAIN_SOURCE_DIR}/temp/tbb/lib/libtbb.so.2
        ${LLVM_BINARY_DIR}/lib/libmlir_runner_utils.so.12git
        ${LLVM_BINARY_DIR}/lib/libomp.so
    DESTINATION bin/lib
    COMPONENT BENCHMARK_APP
)

install(
    DIRECTORY ${IE_MAIN_SOURCE_DIR}/temp/opencv_4.5.0_ubuntu20/opencv/lib/
    DESTINATION bin/lib
    COMPONENT BENCHMARK_APP
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING
    PATTERN "libopencv_core.so*"
    PATTERN "libopencv_img*.so*"
)
