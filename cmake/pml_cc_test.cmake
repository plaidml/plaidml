# Copyright 2020 Intel Corporation

# Heavily inspired by and with gratitude to the IREE project:
# https://github.com/google/iree/blob/main/build_tools/cmake/iree_cc_test.cmake

include(CMakeParseArguments)

# pml_cc_test()
#
# CMake function to imitate Bazel's cc_test rule.
#
# Parameters:
# NAME: name of target (see Usage below)
# SRCS: List of source files for the binary
# DATA: List of other targets and files required for this binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
#
# Note:
# By default, pml_cc_test will always create a binary named pml_${NAME}.
# This will also add it to ctest list as pml_${NAME}.
#
# Usage:
# pml_cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
#   PUBLIC
# )
#
# pml_cc_test(
#   NAME
#     awesome_test
#   SRCS
#     "awesome_test.cc"
#   DEPS
#     gtest_main
#     pml::awesome
# )
function(pml_cc_test)
  if(NOT PML_BUILD_TESTS OR IS_SUBPROJECT)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS;LABELS"
    ${ARGN}
  )

  pml_package_ns(_PACKAGE_NS)
  # Replace dependencies passed by ::name with ::pml::package::name
  list(TRANSFORM _RULE_DATA REPLACE "^::" "${_PACKAGE_NS}::")
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  # Prefix the library with the package name, so we get: pml_package_name
  pml_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  add_executable(${_NAME} "")
  target_sources(${_NAME} PRIVATE ${_RULE_SRCS})
  target_include_directories(${_NAME} SYSTEM PUBLIC ${PML_COMMON_INCLUDE_DIRS})
  target_compile_definitions(${_NAME} PUBLIC ${_RULE_DEFINES})
  target_compile_options(${_NAME} PRIVATE ${_RULE_COPTS})
  target_link_options(${_NAME}
    PRIVATE
      ${_RULE_LINKOPTS}
      ${PML_DEFAULT_LINKOPTS}
  )
  pml_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA})
  # Add all PML targets to a folder in the IDE for organization.
  set_property(TARGET ${_NAME} PROPERTY FOLDER ${PML_IDE_FOLDER}/test)

  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${PML_CXX_STANDARD})
  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

  # Defer computing transitive dependencies and calling target_link_libraries()
  # until all libraries have been declared.
  # Track target and deps, use in pml_complete_binary_link_options() later.
  list(APPEND _RULE_DEPS gmock)
  set_property(GLOBAL APPEND PROPERTY _PML_CC_BINARY_NAMES "${_NAME}")
  set_property(TARGET ${_NAME} PROPERTY DIRECT_DEPS ${_RULE_DEPS})

  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(_TEST_NAME "${_PACKAGE_PATH}/${_RULE_NAME}")

  add_test(
    NAME ${_TEST_NAME}
    COMMAND "$<TARGET_FILE:${_NAME}>"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  )

  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")
  set_property(TEST ${_TEST_NAME} PROPERTY LABELS "${_RULE_LABELS}")

  add_dependencies(check-test ${_NAME})
endfunction()
