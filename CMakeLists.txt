cmake_minimum_required(VERSION 3.13.4)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

include(FetchContent)

# set the project name
project(PlaidML LANGUAGES CXX C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(PML_IDE_FOLDER PML)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#-------------------------------------------------------------------------------
# Project component configuration
#-------------------------------------------------------------------------------
option(PML_BUILD_TESTS "Builds PML unit tests." ON)

list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/cmake
  ${CMAKE_CURRENT_LIST_DIR}/cmake/third_party
)

#-------------------------------------------------------------------------------
# Third party dependencies
#-------------------------------------------------------------------------------
include(easylogging)
include(googletest)
include(llvm-project)

#-------------------------------------------------------------------------------
# cmake modules
#-------------------------------------------------------------------------------
include(pml_copts)
include(pml_macros)
include(pml_cc_binary)
include(pml_cc_library)
include(pml_tblgen_library)
include(pml_whole_archive_link)
include(pml_lit_test)

if(${PML_BUILD_TESTS})
  enable_testing(pmlc)
endif()

add_subdirectory(pmlc)

# Note: this must be called after all libraries have been declared.
pml_complete_binary_link_options()
