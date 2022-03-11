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

set(src_plugins_file "$<TARGET_FILE_DIR:${TARGET_NAME}>/plugins.xml")
set(dest_plugins_dir "${OpenvinoInstall_DIR}/lib")
add_custom_command(
    TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "${src_plugins_file}" "${dest_plugins_dir}"
    COMMENT "Copying ${src_plugins_file} to build directory"
)
