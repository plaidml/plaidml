project("Devkit-Test")

cmake_minimum_required(VERSION 3.18)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(PLAIDML_INSTALL_DIR /path/to/PlaidML)

# Include PlaidML package
include("${PLAIDML_INSTALL_DIR}/share/cmake/PlaidMLConfig.cmake")

# Include 3rd party packages necessary for edsl_tests
include(FetchContent)
include("${PLAIDML_INSTALL_DIR}/share/cmake/third_party/gflags.cmake")
include("${PLAIDML_INSTALL_DIR}/share/cmake/third_party/googletest.cmake")
include("${PLAIDML_INSTALL_DIR}/share/cmake/third_party/half.cmake")
# TODO: Remove LLVM dep for edsl_test
include("${PLAIDML_INSTALL_DIR}/share/cmake/third_party/llvm-project.cmake")

add_executable(edsl_test ${PLAIDML_INSTALL_DIR}/example_project/gtest_main.cc)
target_link_libraries(edsl_test PlaidML::plaidml_plaidml
                                gflags
                                gtest
                                gmock
                                half
                                LLVMSupport)
target_sources(edsl_test PUBLIC ${PLAIDML_INSTALL_DIR}/example_project/edsl_test.cc)

target_include_directories(edsl_test PUBLIC "${PLAIDML_INSTALL_DIR}/include")