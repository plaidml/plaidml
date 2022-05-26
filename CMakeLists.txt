cmake_minimum_required(VERSION 3.13.4 FATAL_ERROR)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
endif()

include(FetchContent)

project(PlaidML LANGUAGES C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(PML_IDE_FOLDER PML)
set(PLAIDML_VERSION 1.0.0)
set(MAIN_BRANCH plaidml-v1)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
  # Support for ninja on windows
  enable_language(ASM)
  add_compile_definitions(
    _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
    NOMINMAX
    WIN32_LEAN_AND_MEAN
  )
endif()

# -------------------------------------------------------------------------------
# Project component configuration
# -------------------------------------------------------------------------------
option(PML_BUILD_TESTS "Builds PML unit tests" ON)
option(PML_ENABLE_DEVKIT "Enable the devkit" ON)
option(PML_INCLUDE_LLVM "Fetch and include the llvm-project" ON)
option(PML_OPENVINO_BRIDGE "Enable the OpenVINO plugin" OFF)
option(PML_ENABLE_ASSERTIONS "Enable assertions" OFF)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  option(PML_ENABLE_ASSERTIONS "Enable assertions" OFF)
else()
  option(PML_ENABLE_ASSERTIONS "Enable assertions" ON)
endif()

# Enable assertion LLVM style.
if(PML_ENABLE_ASSERTIONS)
  # MSVC doesn't like _DEBUG on release builds.
  if(NOT MSVC)
    add_definitions(-D_DEBUG)
  endif()
  # On Release builds cmake automatically defines NDEBUG, so we
  # explicitly undefine it:
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_definitions( -UNDEBUG )
  endif()
else()
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    add_definitions( -DNDEBUG )
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/cmake
  ${CMAKE_CURRENT_LIST_DIR}/cmake/third_party
)

# -------------------------------------------------------------------------------
# Third party dependencies
# -------------------------------------------------------------------------------
if(PML_INCLUDE_LLVM)
  include(llvm-project)
endif()
include(boost)
include(easylogging)
if(NOT InferenceEngineDeveloperPackage_FOUND)
  include(gflags)
endif()
include(googletest)
include(pybind11)
include(mlperf)
include(xsmm)

add_subdirectory(vendor)

# -------------------------------------------------------------------------------
# Enable testing
# -------------------------------------------------------------------------------
if(PML_BUILD_TESTS)
  enable_testing()
endif()

# -------------------------------------------------------------------------------
# cmake modules
# -------------------------------------------------------------------------------
include(pml_format)
include(pml_copts)
include(pml_macros)
include(pml_cc_binary)
include(pml_cc_library)
include(pml_cc_test)
include(pml_lit_test)
include(pml_python)
include(pml_tblgen_library)
include(pml_whole_archive_link)

get_directory_property(IS_SUBPROJECT PARENT_DIRECTORY)

# Set general cpack variables for package description
set(CPACK_GENERATOR "TGZ")
set(CPACK_ARCHIVE_COMPONENT_INSTALL 1)
set(CPACK_PACKAGE_NAME "PlaidML")
set(CPACK_PACKAGE_VENDOR "Intel Corp")
set(CPACK_PACKAGE_CONTACT "Intel")
set(CPACK_PACKAGE_VERSION ${PLAIDML_VERSION})
set(CPACK_COMPONENTS_IGNORE_GROUPS 1)
set(CPACK_COMPONENTS_ALL
  devkit
  testkit
)
include(CPack)


add_subdirectory(plaidbench)
add_subdirectory(plaidml)
add_subdirectory(pmlc)
add_subdirectory(docs)
add_subdirectory(networks)
add_subdirectory(mlperf)

# Note: this must be called after all libraries have been declared.
pml_complete_binary_link_options()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

if(PML_ENABLE_DEVKIT)
  include(devkit)
endif()

if(PML_BUILD_TESTS)
  get_property(_CHECKS GLOBAL PROPERTY GLOBAL_CHECKS)
  foreach(_CHECK ${_CHECKS})
    get_property(_CHECK_DEPS GLOBAL PROPERTY GLOBAL_CHECK_DEPS_${_CHECK})
    add_custom_target(check-${_CHECK}
      COMMAND ${CMAKE_CTEST_COMMAND} -L ${_CHECK} --output-on-failure
      USES_TERMINAL
      DEPENDS ${_CHECK_DEPS}
    )
  endforeach()
endif()
