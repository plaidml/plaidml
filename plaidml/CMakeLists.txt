add_subdirectory(core)
add_subdirectory(edsl)
add_subdirectory(exec)
add_subdirectory(op)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
include(symbols)

pml_cc_library(
  NAME plaidml
  TYPE SHARED
  LINKOPTS
    # TODO: use platform-specific flags
    -Wl,--version-script ${LINUX_LDS}
  PROPS
    LINK_DEPENDS ${LINUX_LDS}
    LINK_DEPENDS ${MACOS_LD}
    LINK_DEPENDS ${WINDOWS_DEF}
    OUTPUT_NAME plaidml
    SOVERSION 1
  DEPS
    plaidml::core::obj
    plaidml::edsl::obj
    plaidml::exec::obj
    plaidml::op::obj
    pmlc::rt::llvm
    pmlc::target::x86
)

pml_cc_library(
  NAME testenv
  HDRS testenv.h
  SRCS testenv.cc
  DEPS
    ::plaidml
    gflags
    gmock
    gtest
    half
)
