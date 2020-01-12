# PlaidML CMake Package Configuration

set(PLAIDML_FOUND TRUE)
get_filename_component(PLAIDML_INCLUDE_DIRS ${PlaidML_DIR}/../../include REALPATH)
get_filename_component(PLAIDML_LIBRARIES ${PlaidML_DIR}/../../lib/${CMAKE_SHARED_LIBRARY_PREFIX}plaidml${CMAKE_SHARED_LIBRARY_SUFFIX} REALPATH)
