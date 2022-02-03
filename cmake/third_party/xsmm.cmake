message("Fetching xsmm")
FetchContent_Declare(
  xsmm
  URL      https://github.com/libxsmm/libxsmm/archive/011dfb33d01e97274f70aac73f40eb68957f6e32.zip
  URL_HASH SHA256=a1a36ab2421d6d433f15b5656cef20126f066cdc7e95cb6354fbdfdf49027695
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
