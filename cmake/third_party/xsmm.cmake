message("Fetching xsmm")
FetchContent_Declare(
  xsmm
  URL      https://github.com/libxsmm/libxsmm/archive/394294d7f62efdd25033ee46e73abd4ccb3ded59.zip
  URL_HASH SHA256=b2307f0374fac07f74414dc4365e078aa97d7abc59204c7955a658ad4e8a676e
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
