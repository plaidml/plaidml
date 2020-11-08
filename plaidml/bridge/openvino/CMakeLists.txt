# add_subdirectory(ops)
# add_subdirectory(plugin)
# add_subdirectory(tests/functional)

# CPMFindPackage(
#   NAME InferenceEngineDeveloperPackage
#   GITHUB_REPOSITORY plaidml/openvino
#   GIT_TAG a2d0a79535e96ba7ae6f552cbda670501d6f2cad
#   DOWNLOAD_ONLY
#   OPTIONS
#     "ENABLE_CLDNN OFF"
#     "ENABLE_DATA OFF"
#     "ENABLE_FUNCTIONAL_TESTS ON"
#     "ENABLE_GAPI_TESTS OFF"
#     "ENABLE_GNA OFF"
#     "ENABLE_MKL_DNN OFF"
#     "ENABLE_MYRIAD OFF"
#     "ENABLE_PROFILING_ITT OFF"
#     "ENABLE_SPEECH_DEMO OFF"
#     "ENABLE_TESTS ON"
#     "ENABLE_VPU ON"
#     "NGRAPH_EXPORT_TARGETS_ENABLE OFF"
#     "VERBOSE_BUILD ON"
# )

find_package(InferenceEngineDeveloperPackage)
if(NOT InferenceEngineDeveloperPackage_FOUND)
  message("HEY!")
endif()

pml_cc_library(
  NAME openvino
  TYPE STATIC
  HDRS
    ops.hpp
    builder.hpp
    util.hpp
  SRCS
    builder.cpp
    util.cpp
    ops/add.cpp
  DEPS
    IE::inference_engine
    # IE::ngraph
    plaidml::core
    plaidml::edsl
    plaidml::op
)
