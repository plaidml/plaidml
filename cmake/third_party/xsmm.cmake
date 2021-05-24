message("Fetching xsmm")
FetchContent_Declare(
  xsmm
  URL      https://github.com/hfp/libxsmm/archive/8b32567d69a6c0e9e8b982d47c8514185d98eaa8.zip
  URL_HASH SHA256=ba1da6a1f9db48bf1eb2ee77c451cb2a56528008a67f12aa34e638160bf6c9eb
)

FetchContent_GetProperties(xsmm)
if(NOT xsmm_POPULATED)
  FetchContent_Populate(xsmm)
endif()

file(GLOB _GLOB_XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${xsmm_SOURCE_DIR}/src/*.c)
list(REMOVE_ITEM _GLOB_XSMM_SRCS ${xsmm_SOURCE_DIR}/src/libxsmm_generator_gemm_driver.c)

add_library(xsmm STATIC ${_GLOB_XSMM_SRCS})
target_include_directories(xsmm PUBLIC ${xsmm_SOURCE_DIR}/include)
target_compile_definitions(xsmm PRIVATE
  __BLAS=0
  LIBXSMM_DEFAULT_CONFIG
)
