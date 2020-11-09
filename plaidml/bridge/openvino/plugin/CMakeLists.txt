set(TARGET_NAME "plaidmlPlugin")

ie_add_plugin(
  NAME ${TARGET_NAME}
  DEVICE_NAME "PLAIDML"
  SOURCES
    plaidml_executable_network.cpp
    plaidml_executable_network.hpp
    plaidml_infer_request.cpp
    plaidml_infer_request.hpp
    plaidml_plugin.cpp
    plaidml_plugin.hpp
  VERSION_DEFINES_FOR plaidml_plugin.cpp
)

target_link_libraries(${TARGET_NAME}
  PRIVATE
    IE::inference_engine
    ${NGRAPH_LIBRARIES}
)

pml_link_whole_libs(${TARGET_NAME}
  plaidml::bridge::openvino
)

ie_register_plugins(
  MAIN_TARGET ${TARGET_NAME}
  POSSIBLE_PLUGINS ${TARGET_NAME}
)
