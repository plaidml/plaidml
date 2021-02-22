project("Devkit-Test")

cmake_minimum_required(VERSION 3.18)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# MUST SET THIS W/ CORRECT EXTRACTION PATH
set(PlaidML_DIR /path/to/PlaidML-Package/share/PlaidML/cmake)
find_package(PlaidML CONFIG REQUIRED)

add_executable(edsl_test edsl_tests.cc)
target_link_libraries(edsl_test PlaidML::plaidml_edsl_edsl)
target_link_libraries(edsl_test PlaidML::plaidml_exec_exec)
target_link_libraries(edsl_test PlaidML::plaidml_op_op)
target_link_libraries(edsl_test PlaidML::plaidml_testenv)
target_link_libraries(edsl_test PlaidML::pmlc_util_util)
target_link_libraries(edsl_test PlaidML::gmock)
target_link_libraries(edsl_test PlaidML::gtest)
target_link_libraries(edsl_test PlaidML::half)
target_link_libraries(edsl_test PlaidML::LLVMSupport)

# TODO: Make mapping of inclusions in project to actual target names?