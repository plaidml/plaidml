# Copyright 2020 Intel Corporation

include(CMakeParseArguments)

function(pml_lit_test)
  if(NOT PML_BUILD_TESTS OR IS_SUBPROJECT)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "DATA"
    ${ARGN}
  )

  pml_package_ns(_PACKAGE_NS)
  # Replace dependencies passed by ::name with ::pml::package::name
  list(TRANSFORM _RULE_DATA REPLACE "^::" "${_PACKAGE_NS}::")

  # Prefix the library with the package name, so we get: pml_package_name
  pml_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  set(_COMMAND ${LLVM_BINARY_DIR}/bin/llvm-lit ${CMAKE_CURRENT_BINARY_DIR} -v)
  add_custom_target(${_NAME}
    COMMAND ${_COMMAND}
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    USES_TERMINAL
  )
  pml_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA})

  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(_TEST_NAME "${_PACKAGE_PATH}/${_RULE_NAME}")

  add_test(
    NAME ${_TEST_NAME}
    COMMAND ${_COMMAND}
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  set_target_properties(${_NAME} PROPERTIES FOLDER "Tests")
endfunction()
