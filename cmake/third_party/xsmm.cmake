message("Fetching xsmm")
FetchContent_Declare(
  xsmm
  URL      https://github.com/libxsmm/libxsmm/archive/362a48f944ed72f73a1883fd607311823dae3c96.zip
  URL_HASH SHA256=8b8bf575abc53fa76876ddd093e04c6f7b10ecd964777ab83236d0d6352906ea
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
