# Copyright 2020 Intel Corporation

# Heavily inspired by and with gratitude to the IREE project:
# https://github.com/google/iree/blob/main/build_tools/cmake/iree_tablegen_library.cmake

include(CMakeParseArguments)

set(MLIR_TABLEGEN_EXE mlir-tblgen)

# pml_tblgen_library()
#
# Runs tblgen to produce some artifacts.
function(pml_tblgen_library)
  cmake_parse_arguments(
    _RULE
    "TESTONLY"
    "NAME;TBLGEN"
    "TD_FILE;OUTS;OPTS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT PML_BUILD_TESTS)
    return()
  endif()

  # Prefix the library with the package name, so we get: pml_package_name
  pml_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  if(_RULE_TBLGEN)
    set(_TBLGEN "${_RULE_TBLGEN}")
  else()
    set(_TBLGEN "MLIR")
  endif()

  set(LLVM_TARGET_DEFINITIONS ${_RULE_TD_FILE})
  set(_INCLUDE_DIRS ${PML_COMMON_INCLUDE_DIRS})
  list(APPEND _INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
  list(TRANSFORM _INCLUDE_DIRS PREPEND "-I")
  set(_OUTPUTS)
  while(_RULE_OUTS)
    list(POP_FRONT _RULE_OUTS _COMMAND _FILE)
    tablegen(${_TBLGEN} ${_FILE} ${_COMMAND} ${_RULE_OPTS} ${_INCLUDE_DIRS})
    list(APPEND _OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/${_FILE})
  endwhile()
  add_custom_target(${_NAME}_target DEPENDS ${_OUTPUTS})
  set_target_properties(${_NAME}_target PROPERTIES FOLDER "Tablegenning")

  add_library(${_NAME} INTERFACE)
  add_dependencies(${_NAME} ${_NAME}_target)

  # Alias the pml_package_name library to pml::package::name.
  pml_package_ns(_PACKAGE_NS)
  add_library(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})
endfunction()
