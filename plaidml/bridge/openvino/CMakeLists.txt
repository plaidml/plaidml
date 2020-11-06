add_subdirectory(ops)

set(_SRCS
    plaidml_program_builder.cpp
    plaidml_util.cpp)
set(_HDRS
    plaidml_ops.hpp
    plaidml_program_builder.hpp
    plaidml_util.hpp)

# TODO
find_package(InferenceEngineDeveloperPackage REQUIRED)


# TODO For experiment only
# include_directories( /home/tim/openvino/inference-engine/include )
# end TODO experiment




pml_cc_library(
  NAME program_builder
  TYPE STATIC
  HDRS ${_HDRS}
  SRCS ${_SRCS}
  # TODO: Confirm & clean DEPS
  DEPS
    plaidml::core
    # ie_libraries
    # inference_engine
    # IE::inference_engine_preproc
    # IE::inference_engine_plugin_api
    IE::inference_engine
    # ngraph_builders
    IE::ngraph


    # pmlc::ast
    # pmlc::util
    # LLVMSupport
)
