FetchContent_Declare(
  easyloggingpp
  URL      https://github.com/amrayn/easyloggingpp/archive/v9.96.7.tar.gz
  URL_HASH SHA256=237c80072b9b480a9f2942b903b4b0179f65e146e5dcc64864dc91792dedd722
)
FetchContent_MakeAvailable(easyloggingpp)

add_library(easyloggingpp STATIC ${easyloggingpp_SOURCE_DIR}/src/easylogging++.cc)
target_include_directories(easyloggingpp PUBLIC ${easyloggingpp_SOURCE_DIR}/src)
target_compile_definitions(easyloggingpp PRIVATE
  ELPP_THREAD_SAFE
  ELPP_CUSTOM_COUT=std::cerr
  ELPP_STL_LOGGING
  ELPP_LOG_STD_ARRAY
  ELPP_LOG_UNORDERED_MAP
  ELPP_LOG_UNORDERED_SET
  ELPP_NO_LOG_TO_FILE
  ELPP_DISABLE_DEFAULT_CRASH_HANDLING
  ELPP_WINSOCK2
)
set_property(TARGET easyloggingpp PROPERTY POSITION_INDEPENDENT_CODE ON)
