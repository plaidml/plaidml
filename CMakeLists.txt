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

# set the project name
project(PlaidML LANGUAGES CXX C)

include(FetchContent)

# TODO - dedupe from workspace.bzl
SET(LLVM_COMMIT b3f1f66eddd9ed4e3caf6043344b17f5b0920bb0)
#-------------------------------------------------------------------------------
# Third party: llvm/mlir
#-------------------------------------------------------------------------------

set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(LLVM_APPEND_VC_REV OFF CACHE BOOL "" FORCE)
set(LLVM_ENABLE_IDE ON CACHE BOOL "" FORCE)
set(LLVM_ENABLE_RTTI ON CACHE BOOL "" FORCE)
set(LLVM_TARGETS_TO_BUILD "WebAssembly;X86;ARM;AArch64" CACHE STRING "" FORCE)
set(LLVM_ENABLE_PROJECTS "mlir;" CACHE STRING "" FORCE)
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "" FORCE)

set(MLIR_TABLEGEN_EXE mlir-tblgen)

FetchContent_Declare(
  llvm_project
  FETCHCONTENT_QUIET OFF
  URL https://github.com/plaidml/llvm-project/archive/${LLVM_COMMIT}.tar.gz
  URL_HASH SHA256=217fb2d6b249e886d6954612ff65dbf834a418e1f1c835c47445722afb5a54be
)
FetchContent_GetProperties(llvm_project)
if(NOT llvm_project_POPULATED)
  FetchContent_Populate(llvm_project)
  #find_package(MLIR PATHS ${CMAKE_BINARY_DIR}/lib/cmake/mlir/)
  add_subdirectory(${llvm_project_SOURCE_DIR}/llvm ${llvm_project_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

list(APPEND LLVM_INCLUDE_DIRS ${llvm_project_SOURCE_DIR}/llvm/include ${llvm_project_BINARY_DIR}/include)
list(APPEND MLIR_INCLUDE_DIRS ${llvm_project_SOURCE_DIR}/mlir/include ${llvm_project_BINARY_DIR}/tools/mlir/include)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

#############
# Easylogging
#############
FetchContent_Declare(
  easyloggingpp
  FETCHCONTENT_QUIET OFF
  URL https://github.com/amrayn/easyloggingpp/archive/v9.96.7.tar.gz
  URL_HASH SHA256=237c80072b9b480a9f2942b903b4b0179f65e146e5dcc64864dc91792dedd722
)
FetchContent_GetProperties(easyloggingpp)
if(NOT easyloggingpp_POPULATED)
  FetchContent_Populate(easyloggingpp)
  #find_package(MLIR PATHS ${CMAKE_BINARY_DIR}/lib/cmake/mlir/)
  add_subdirectory(${easyloggingpp_SOURCE_DIR} ${easyloggingpp_BINARY_DIR}EXCLUDE_FROM_ALL)
endif()
include_directories(${easyloggingpp_SOURCE_DIR}/src)

#########
# Plaidml
#########

set(PML_IDE_FOLDER PML)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/cmake/
)

include(pml_copts)
include(pml_macros)
include(pml_cc_binary)
include(pml_cc_library)


# C++
list(APPEND PMLC_DEFAULT_COPTS -Wno-c++17-extensions)
# Obviously too broad, can be addressed later
include_directories(./)
include_directories(${CMAKE_BINARY_DIR})

add_subdirectory(pmlc)
