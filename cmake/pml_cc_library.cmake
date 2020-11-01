# Copyright 2020 Intel Corporation

# Heavily inspired by and with gratitude to the IREE project:
# https://github.com/google/iree/blob/main/build_tools/cmake/iree_cc_library.cmake

include(CMakeParseArguments)

# pml_cc_library()
#
# CMake function to imitate Bazel's cc_library rule.
#
# Parameters:
# NAME: name of target (see Note)
# HDRS: List of public header files for the library
# TEXTUAL_HDRS: List of public header files that cannot be compiled on their own
# SRCS: List of source files for the library
# DATA: List of other targets and files required for this binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# INCLUDES: Include directories to add to dependencies
# LINKOPTS: List of link options
# ALWAYSLINK: Always link the library into any binary with a direct dep.
# PUBLIC: Add this so that this library will be exported under pml::
# Also in IDE, target will appear in PML folder while non PUBLIC will be in PML/internal.
# TESTONLY: When added, this target will only be built if user passes -DPML_BUILD_TESTS=ON to CMake.
# WHOLEARCHIVE: If set, links all symbols from "ALWAYSLINK" libraries.
#
# Note:
# By default, pml_cc_library will always create a library named pml_${NAME},
# and alias target pml::${NAME}. The pml:: form should always be used.
# This is to reduce namespace pollution.
#
# pml_cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
# pml_cc_library(
#   NAME
#     fantastic_lib
#   SRCS
#     "b.cc"
#   DEPS
#     pml::package::awesome # not "awesome" !
#   PUBLIC
# )
#
# pml_cc_library(
#   NAME
#     main_lib
#   ...
#   DEPS
#     pml::package::fantastic_lib
# )
function(pml_cc_library)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;ALWAYSLINK;TESTONLY;WHOLEARCHIVE"
    "NAME;TYPE"
    "HDRS;TEXTUAL_HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS;INCLUDES;PROPS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT PML_BUILD_TESTS)
    return()
  endif()

  pml_package_ns(_PACKAGE_NS)
  # Replace dependencies passed by ::name with ::pml::package::name
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
  list(TRANSFORM _RULE_DATA REPLACE "^::" "${_PACKAGE_NS}::")

  # Prefix the library with the package name, so we get: pml_package_name.
  pml_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  # Check if this is a header-only library.
  # Note that as of February 2019, many popular OS's (for example, Ubuntu
  # 16.04 LTS) only come with cmake 3.5 by default.  For this reason, we can't
  # use list(FILTER...)
  set(_CC_SRCS "${_RULE_SRCS}")
  foreach(src_file IN LISTS _CC_SRCS)
    if(${src_file} MATCHES ".*\\.(h|inc)")
      list(REMOVE_ITEM _CC_SRCS "${src_file}")
    endif()
  endforeach()

  if(NOT _RULE_TYPE)
    if("${_CC_SRCS}" STREQUAL "")
      set(_RULE_TYPE INTERFACE)
    else()
      set(_RULE_TYPE STATIC)
    endif()
  endif()

  if (${_RULE_TYPE} STREQUAL SHARED)
    if (_RULE_WHOLEARCHIVE)
      message(FATAL_ERROR "WHOLEARCHIVE must be set together with SHARED")
    endif()
  endif()

  if(NOT ${_RULE_TYPE} STREQUAL INTERFACE)
    add_library(${_NAME} ${_RULE_TYPE} "")

    target_sources(${_NAME}
      PRIVATE
        ${_RULE_SRCS}
        ${_RULE_TEXTUAL_HDRS}
        ${_RULE_HDRS}
    )
    target_include_directories(${_NAME} SYSTEM
      PUBLIC
        "$<BUILD_INTERFACE:${PML_COMMON_INCLUDE_DIRS}>"
    )
    target_include_directories(${_NAME}
      PUBLIC
        "$<BUILD_INTERFACE:${_RULE_INCLUDES}>"
    )
    target_compile_options(${_NAME}
      PRIVATE
        ${_RULE_COPTS}
        ${PML_DEFAULT_COPTS}
    )
    if(_RULE_WHOLEARCHIVE)
      pml_whole_archive_link(${_NAME} ${_RULE_DEPS})
    else()
      target_link_libraries(${_NAME} PUBLIC ${_RULE_DEPS})
    endif()
    target_link_libraries(${_NAME}
      PRIVATE
        ${_RULE_LINKOPTS}
        ${PML_DEFAULT_LINKOPTS}
    )

    pml_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA})
    target_compile_definitions(${_NAME}
      PUBLIC
        ${_RULE_DEFINES}
    )

    if(DEFINED _RULE_ALWAYSLINK)
      set_property(TARGET ${_NAME} PROPERTY ALWAYSLINK 1)
    endif()

    # Add all PML targets to a folder in the IDE for organization.
    if(_RULE_PUBLIC)
      set_property(TARGET ${_NAME} PROPERTY FOLDER ${PML_IDE_FOLDER})
    elseif(_RULE_TESTONLY)
      set_property(TARGET ${_NAME} PROPERTY FOLDER ${PML_IDE_FOLDER}/test)
    else()
      set_property(TARGET ${_NAME} PROPERTY FOLDER ${PML_IDE_FOLDER}/internal)
    endif()

    # INTERFACE libraries can't have the CXX_STANDARD property set.
    set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${PML_CXX_STANDARD})
    set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
  else()
    # Generating header-only library.
    add_library(${_NAME} INTERFACE)
    target_include_directories(${_NAME} SYSTEM
      INTERFACE
        "$<BUILD_INTERFACE:${PML_COMMON_INCLUDE_DIRS}>"
    )
    target_compile_options(${_NAME}
      INTERFACE
        ${_RULE_COPTS}
        ${PML_DEFAULT_COPTS}
    )
    target_link_libraries(${_NAME}
      INTERFACE
        ${_RULE_DEPS}
        ${_RULE_LINKOPTS}
        ${PML_DEFAULT_LINKOPTS}
    )
    pml_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA})
    target_compile_definitions(${_NAME}
      INTERFACE
        ${_RULE_DEFINES}
    )
  endif()

  if (_RULE_PROPS)
    set_target_properties(${_NAME} PROPERTIES ${_RULE_PROPS})
  endif()

  # Alias the pml_package_name library to pml::package::name.
  # This lets us more clearly map to Bazel and makes it possible to
  # disambiguate the underscores in paths vs. the separators.
  add_library(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})
  pml_package_dir(_PACKAGE_DIR)
  if(${_RULE_NAME} STREQUAL ${_PACKAGE_DIR})
    # If the library name matches the package then treat it as a default.
    # For example, foo/bar/ library 'bar' would end up as 'foo::bar'.
    add_library(${_PACKAGE_NS} ALIAS ${_NAME})
  endif()
endfunction()
