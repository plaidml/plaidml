# Copyright 2020 Intel Corporation

# Heavily inspired by and with gratitude to the IREE project:
# https://github.com/google/iree/blob/main/build_tools/cmake/iree_cc_binary.cmake

include(CMakeParseArguments)

if (NOT DEFINED _PML_CC_BINARY_NAMES)
  set(_PML_CC_BINARY_NAMES "")
endif()

# pml_cc_binary()
#
# CMake function to imitate Bazel's cc_binary rule.
#
# Parameters:
# NAME: name of target (see Usage below)
# SRCS: List of source files for the binary
# DATA: List of other targets and files required for this binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
# TESTONLY: for testing; won't compile when tests are disabled
# HOSTONLY: host only; compile using host toolchain when cross-compiling
#
# Note:
# By default, pml_cc_binary will always create a binary named pml_${NAME}.
#
# pml_cc_binary(
#   NAME
#     awesome_tool
#   OUT
#     awesome-tool
#   SRCS
#     "awesome-tool-main.cc"
#   DEPS
#     pml::awesome
# )
function(pml_cc_binary)
  cmake_parse_arguments(
    _RULE
    "HOSTONLY;TESTONLY"
    "NAME;OUT"
    "SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT PML_BUILD_TESTS)
    return()
  endif()

  # Prefix the library with the package name, so we get: pml_package_name
  pml_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  if(_RULE_HOSTONLY AND CMAKE_CROSSCOMPILING)
    # The binary is marked as host only. We need to declare the rules for
    # generating them under host configuration so when cross-compiling towards
    # target we can still have this binary.
    pml_declare_host_excutable(${_RULE_NAME} ${_NAME})

    # Still define the package-prefixed target so we can have a consistent way
    # to reference this binary, whether cross-compiling or not. But this time
    # use the target to convey a property for the executable path under host
    # configuration.
    pml_get_executable_path(_EXE_PATH ${_RULE_NAME})
    add_custom_target(${_NAME} DEPENDS ${_EXE_PATH})
    set_target_properties(${_NAME} PROPERTIES HOST_TARGET_FILE "${_EXE_PATH}")
    return()
  endif()

  add_executable(${_NAME} "")
  add_executable(${_RULE_NAME} ALIAS ${_NAME})
  if(_RULE_SRCS)
    target_sources(${_NAME}
      PRIVATE
        ${_RULE_SRCS}
    )
  else()
    set(_DUMMY_SRC "${CMAKE_CURRENT_BINARY_DIR}/${_NAME}_dummy.cc")
    file(WRITE ${_DUMMY_SRC} "")
    target_sources(${_NAME}
      PRIVATE
        ${_DUMMY_SRC}
    )
  endif()
  if(_RULE_OUT)
    set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_RULE_OUT}")
  else()
    set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_RULE_NAME}")
  endif()
  target_include_directories(${_NAME} SYSTEM
    PUBLIC
      ${PML_COMMON_INCLUDE_DIRS}
  )
  target_compile_definitions(${_NAME}
    PUBLIC
      ${_RULE_DEFINES}
  )
  target_compile_options(${_NAME}
    PRIVATE
      ${_RULE_COPTS}
  )
  target_link_options(${_NAME}
    PRIVATE
      ${_RULE_LINKOPTS}
      ${PML_DEFAULT_LINKOPTS}
  )
  pml_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA})

  pml_package_ns(_PACKAGE_NS)
  # Replace dependencies passed by ::name with ::pml::package::name
  list(TRANSFORM _RULE_DATA REPLACE "^::" "${_PACKAGE_NS}::")
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  # Add all PML targets to a folder in the IDE for organization.
  set_property(TARGET ${_NAME} PROPERTY FOLDER ${PML_IDE_FOLDER}/binaries)

  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${PML_CXX_STANDARD})
  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

  # Defer computing transitive dependencies and calling target_link_libraries()
  # until all libraries have been declared.
  # Track target and deps, use in pml_complete_binary_link_options() later.
  set_property(GLOBAL APPEND PROPERTY _PML_CC_BINARY_NAMES "${_NAME}")
  set_property(TARGET ${_NAME} PROPERTY DIRECT_DEPS ${_RULE_DEPS})

  install(TARGETS ${_NAME}
          RENAME ${_RULE_NAME}
          COMPONENT ${_RULE_NAME}
          RUNTIME DESTINATION bin)
endfunction()

# Sets target_link_libraries() on all registered binaries.
# This must be called after all libraries have been declared.
function(pml_complete_binary_link_options)
  get_property(_NAMES GLOBAL PROPERTY _PML_CC_BINARY_NAMES)

  foreach(_NAME ${_NAMES})
    get_target_property(_DIRECT_DEPS ${_NAME} DIRECT_DEPS)
    pml_whole_archive_link(${_NAME} ${_DIRECT_DEPS})
  endforeach(_NAME)
endfunction()
