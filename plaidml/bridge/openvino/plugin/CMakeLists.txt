set(TARGET_NAME "plaidmlPlugin")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "PLAIDML"
              SOURCES ${SOURCE} ${HEADERS}
              VERSION_DEFINES_FOR plaidml_plugin.cpp)

# TODO: Unclear if this include dirs is needed
target_include_directories(${TARGET_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(${TARGET_NAME} PRIVATE IE::inference_engine ${NGRAPH_LIBRARIES} plaidml::core plaidml::bridge::openvino::program_builder)

ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
                    POSSIBLE_PLUGINS ${TARGET_NAME})
