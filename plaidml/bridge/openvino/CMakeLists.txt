add_subdirectory(ops)

find_package(InferenceEngineDeveloperPackage REQUIRED)

# TODO For experiment only
# include_directories( /home/tim/openvino/inference-engine/include )
# end TODO experiment


pml_cc_library(
  NAME program_builder
  TYPE STATIC
  HDRS
    plaidml_ops.hpp
    plaidml_program_builder.hpp
    plaidml_util.hpp
  SRCS
    plaidml_program_builder.cpp
    plaidml_util.cpp
  DEPS
    IE::inference_engine
    IE::ngraph
    plaidml::core
)

# TODO: Plugin stuff
add_subdirectory(plugin)

# TODO: Test stuff
add_subdirectory(tests/functional)
