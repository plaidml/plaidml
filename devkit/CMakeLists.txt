project("PlaidML-Devkit")

cmake_minimum_required(VERSION 3.18)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

find_package(PlaidML REQUIRED PATHS "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Include 3rd party packages necessary for edsl_tests
include(FetchContent)
include("cmake/gflags.cmake")
include("cmake/googletest.cmake")
include("cmake/half.cmake")

add_executable(edsl_test
  src/edsl_test.cc
  src/testenv.cc
)

target_link_libraries(edsl_test
  PlaidML
  gflags
  gtest_main
  gmock
  half
)

target_include_directories(edsl_test PUBLIC include)
