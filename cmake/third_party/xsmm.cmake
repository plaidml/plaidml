# Use LIBXSMM (make PREFIX=/path/to/libxsmm) given by LIBXSMMROOT
set(LIBXSMMROOT $ENV{LIBXSMMROOT})
# Fetch LIBXSMM (even if LIBXSMMROOT is present)
set(LIBXSMMFETCH $ENV{LIBXSMMFETCH})

if(LIBXSMMROOT AND NOT LIBXSMMFETCH)
  message(STATUS "Found LIBXSMM (${LIBXSMMROOT})")
  file(GLOB XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${LIBXSMMROOT}/include/libxsmm/*.c)
  list(REMOVE_ITEM XSMM_SRCS ${LIBXSMMROOT}/include/libxsmm/libxsmm_generator_gemm_driver.c)
else()
  message(STATUS "Fetching LIBXSMM")
  include(FetchContent)

  FetchContent_Declare(
    xsmm
    URL https://github.com/libxsmm/libxsmm/archive/9a0c707384f76bf4a304a4998be9c6c5e7c41400.tar.gz
    URL_HASH SHA256=bc17bc0d07e4210ec4cdb7790a67cf1226c93dc82862e5d89d93e83d81233532
  )

  FetchContent_GetProperties(xsmm)
  if(NOT xsmm_POPULATED)
    FetchContent_Populate(xsmm)
  endif()

  set(LIBXSMMROOT ${xsmm_SOURCE_DIR})
  file(GLOB XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${LIBXSMMROOT}/src/*.c)
  list(REMOVE_ITEM XSMM_SRCS ${LIBXSMMROOT}/src/libxsmm_generator_gemm_driver.c)
endif()
set(XSMM_INCLUDE_DIRS ${LIBXSMMROOT}/include)

add_library(xsmm STATIC ${XSMM_SRCS})
target_include_directories(xsmm PUBLIC ${XSMM_INCLUDE_DIRS})
target_compile_definitions(xsmm PUBLIC
  LIBXSMM_DEFAULT_CONFIG
)
target_compile_definitions(xsmm PRIVATE
  __BLAS=0
)
add_definitions(-U_DEBUG)

target_link_libraries(xsmm PUBLIC pthread)
target_link_libraries(xsmm PUBLIC rt)
target_link_libraries(xsmm PUBLIC dl)
target_link_libraries(xsmm PUBLIC m)
target_link_libraries(xsmm PUBLIC c)
