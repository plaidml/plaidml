FetchContent_Declare(
  xsmm
  URL      https://github.com/hfp/libxsmm/archive/592d72f2902fa49fda51a4221e83f7f978936a9d.zip
  URL_HASH SHA256=2595c625e9e63099de8bea4876a7e5835f0c5f560049db6de31611da2539d448
)
FetchContent_MakeAvailable(xsmm)

file(GLOB _GLOB_XSMM_SRCS LIST_DIRECTORIES false ${xsmm_SOURCE_DIR}/src/*.c)
list(REMOVE_ITEM _GLOB_XSMM_SRCS ${xsmm_SOURCE_DIR}/src/libxsmm_generator_gemm_driver.c)

add_library(xsmm STATIC ${_GLOB_XSMM_SRCS})
target_include_directories(xsmm PUBLIC ${xsmm_SOURCE_DIR}/include)
target_compile_definitions(xsmm PRIVATE
  __BLAS=0
  LIBXSMM_DEFAULT_CONFIG
)
