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
set(LIBOMP_OMPD_SUPPORT OFF CACHE BOOL "" FORCE)
set(OPENMP_ENABLE_LIBOMPTARGET OFF CACHE BOOL "" FORCE)
set(OPENMP_ENABLE_OMPT_TOOLS OFF CACHE BOOL "" FORCE)
set(OPENMP_STANDALONE_BUILD ON CACHE BOOL "" FORCE)

list(APPEND LLVM_EXTERNAL_PROJECTS mlir_hlo)
set(LLVM_EXTERNAL_MLIR_HLO_SOURCE_DIR ${CMAKE_SOURCE_DIR}/vendor/mlir-hlo)

if(LOCAL_LLVM_DIR)
  message("LOCAL_LLVM_DIR: ${LOCAL_LLVM_DIR}")
  set(LLVM_SOURCE_DIR ${LOCAL_LLVM_DIR})
  set(LLVM_BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/llvm-project-build)
  set(LLVM_EXTERNAL_MLIR_SOURCE_DIR "${LLVM_SOURCE_DIR}/mlir")
  add_subdirectory(${LLVM_SOURCE_DIR}/llvm ${LLVM_BINARY_DIR} EXCLUDE_FROM_ALL)
else()
  message(STATUS "Fetching LLVM")
  FetchContent_Declare(
    llvm-project
    URL https://github.com/plaidml/llvm-project/archive/3b8d5b76a2ae3c8fbe65ade73deb0ba7134e0072.tar.gz
    URL_HASH SHA256=bcfcc418da3d4c0df95124674af071cbf8b3128e6f3500831868b735c448f84c
  )
  #FetchContent_MakeAvailable(llvm-project)
  # Check if population has already been performed
  FetchContent_GetProperties(llvm-project)
  if(NOT llvm-project_POPULATED)
    message(STATUS "Populate LLVM") 
    # Fetch the content using previously declared details
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
  ${LLVM_EXTERNAL_MLIR_HLO_SOURCE_DIR}/include
  ${LLVM_BINARY_DIR}/tools/mlir_hlo/include
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
