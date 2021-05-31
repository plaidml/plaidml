set(BUILD_SHARED_LIBS OFF)
set(CROSS_TOOLCHAIN_FLAGS_ "" CACHE STRING "" FORCE)
set(CROSS_TOOLCHAIN_FLAGS_NATIVE "" CACHE STRING "" FORCE)
set(LLVM_APPEND_VC_REV OFF CACHE BOOL "" FORCE)
set(LLVM_ENABLE_IDE OFF CACHE BOOL "" FORCE)
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "" FORCE)
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "" FORCE)
if(CMAKE_CROSSCOMPILING)
  set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
else()
  set(LLVM_ENABLE_PROJECTS "mlir;openmp" CACHE STRING "" FORCE)
endif()
set(LLVM_ENABLE_RTTI ON CACHE BOOL "" FORCE)
set(LLVM_ENABLE_WARNINGS OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_TOOLS ON CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(LLVM_TARGETS_TO_BUILD "X86" CACHE STRING "" FORCE)
set(LIBOMP_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(OPENMP_ENABLE_LIBOMPTARGET OFF CACHE BOOL "" FORCE)
set(OPENMP_ENABLE_OMPT_TOOLS OFF CACHE BOOL "" FORCE)
set(OPENMP_STANDALONE_BUILD ON CACHE BOOL "" FORCE)

if(LOCAL_LLVM_DIR)
  message("LOCAL_LLVM_DIR: ${LOCAL_LLVM_DIR}")
  set(LLVM_SOURCE_DIR ${LOCAL_LLVM_DIR})
  set(LLVM_BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/llvm-project-build)
  set(LLVM_EXTERNAL_MLIR_SOURCE_DIR "${LLVM_SOURCE_DIR}/mlir")
  add_subdirectory(${LLVM_SOURCE_DIR}/llvm ${LLVM_BINARY_DIR} EXCLUDE_FROM_ALL)
else()
  message("Fetching LLVM")
  FetchContent_Declare(
    llvm-project
    URL      https://github.com/plaidml/llvm-project/archive/b8b6faf202164a09169a61783edd61a53d98c9f3.tar.gz
    # URL_HASH SHA256=63399591e570816b706e4f8a9fd25af5b4561aad8e22813ad21dd33fec8c8f43
  )
  FetchContent_GetProperties(llvm-project)
  if(NOT llvm-project_POPULATED)
    FetchContent_Populate(llvm-project)
    set(LLVM_SOURCE_DIR ${llvm-project_SOURCE_DIR})
    set(LLVM_BINARY_DIR ${llvm-project_BINARY_DIR})
    add_subdirectory(${LLVM_SOURCE_DIR}/llvm ${LLVM_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endif()

set(MLIR_SOURCE_DIR "${LLVM_SOURCE_DIR}/mlir")

list(APPEND LLVM_INCLUDE_DIRS
  ${LLVM_SOURCE_DIR}/llvm/include
  ${LLVM_BINARY_DIR}/include
)

list(APPEND MLIR_INCLUDE_DIRS
  ${MLIR_SOURCE_DIR}/include
  ${LLVM_BINARY_DIR}/tools/mlir/include
)

set(LLVM_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS})
set(MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIRS})

include_directories(SYSTEM
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${LIBOMP_INCLUDE_DIR}
)

if(NOT CMAKE_CROSSCOMPILING)
  target_include_directories(omp PUBLIC
    ${LLVM_SOURCE_DIR}/openmp/runtime/src
    ${LLVM_BINARY_DIR}/projects/openmp/runtime/src
  )
endif()
