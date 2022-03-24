set(TARGET_NAME "plaidmlPlugin")

ie_cpack_set_library_dir()

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
    IE::inference_engine_transformations
    ${NGRAPH_LIBRARIES}
    plaidml::bridge::openvino
    plaidml::plaidml
)

ie_register_plugins_dynamic(
  MAIN_TARGET ${TARGET_NAME}
  POSSIBLE_PLUGINS ${TARGET_NAME}
)

add_custom_command(
  TARGET ${TARGET_NAME}
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/plugins.xml
    ${CMAKE_BINARY_DIR}/${BIN_FOLDER}/lib/plugins.xml
)
