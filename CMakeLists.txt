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

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

SET(LLVM_COMMIT f00f71b2a279449c04796e28b3b7bb0e06c945bf)
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
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "" FORCE)

set(MLIR_TABLEGEN_EXE mlir-tblgen)

FetchContent_Declare(
  llvm_project
  FETCHCONTENT_QUIET OFF
  URL https://github.com/plaidml/llvm-project/archive/${LLVM_COMMIT}.tar.gz
  URL_HASH SHA256=88fccbb163aac6192d572ce3ed1538a2f6e4eb9341c529ecccba502bb2f0d144
)
FetchContent_GetProperties(llvm_project)
if(NOT llvm_project_POPULATED)
  FetchContent_Populate(llvm_project)
  #find_package(MLIR PATHS ${CMAKE_BINARY_DIR}/lib/cmake/mlir/)
  add_subdirectory(${llvm_project_SOURCE_DIR}/llvm ${llvm_project_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
list(APPEND CMAKE_MODULE_PATH "${llvm_project_SOURCE_DIR}/llvm/cmake/modules")

#set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
#set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
#set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
list(APPEND LLVM_INCLUDE_DIRS ${llvm_project_SOURCE_DIR}/llvm/include)
list(APPEND MLIR_INCLUDE_DIRS ${llvm_project_SOURCE_DIR}/mlir/include ${llvm_project_BINARY_DIR}/tools/mlir/include)


include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

# link_directories(${LLVM_BUILD_LIBRARY_DIR})
# add_definitions(${LLVM_DEFINITIONS})
include_directories(./)
include_directories(${CMAKE_BINARY_DIR})
add_subdirectory(pmlc)
