message("Fetching xsmm")
FetchContent_Declare(
  xsmm
  URL      https://github.com/hfp/libxsmm/archive/refs/heads/master.zip
)

FetchContent_GetProperties(xsmm)
if(NOT xsmm_POPULATED)
  FetchContent_Populate(xsmm)
endif()

file(GLOB _GLOB_XSMM_SRCS LIST_DIRECTORIES false ${xsmm_SOURCE_DIR}/src/*.c)
list(REMOVE_ITEM _GLOB_XSMM_SRCS ${xsmm_SOURCE_DIR}/src/libxsmm_generator_gemm_driver.c)

add_library(xsmm STATIC ${_GLOB_XSMM_SRCS})
target_include_directories(xsmm PUBLIC ${xsmm_SOURCE_DIR}/include)
target_compile_definitions(xsmm PRIVATE
  __BLAS=0
  LIBXSMM_DEFAULT_CONFIG
)
