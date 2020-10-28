FetchContent_Declare(
  llvm-project
  URL      https://github.com/plaidml/llvm-project/archive/b3f1f66eddd9ed4e3caf6043344b17f5b0920bb0.tar.gz
  URL_HASH SHA256=217fb2d6b249e886d6954612ff65dbf834a418e1f1c835c47445722afb5a54be
)

set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(LLVM_APPEND_VC_REV OFF CACHE BOOL "" FORCE)
set(LLVM_ENABLE_IDE OFF CACHE BOOL "" FORCE)
set(LLVM_ENABLE_RTTI ON CACHE BOOL "" FORCE)
set(LLVM_TARGETS_TO_BUILD "X86" CACHE STRING "" FORCE)
set(LLVM_ENABLE_PROJECTS "mlir;" CACHE STRING "" FORCE)
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "" FORCE)

FetchContent_GetProperties(llvm-project)
if(NOT llvm-project_POPULATED)
  FetchContent_Populate(llvm-project)
  add_subdirectory(${llvm-project_SOURCE_DIR}/llvm ${llvm-project_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

list(APPEND LLVM_INCLUDE_DIRS
  ${llvm-project_SOURCE_DIR}/llvm/include
  ${llvm-project_BINARY_DIR}/include
)

list(APPEND MLIR_INCLUDE_DIRS
  ${llvm-project_SOURCE_DIR}/mlir/include
  ${llvm-project_BINARY_DIR}/tools/mlir/include
)

add_library(LLVM INTERFACE)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
