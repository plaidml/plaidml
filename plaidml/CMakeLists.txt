add_subdirectory(bridge)
add_subdirectory(core)
add_subdirectory(edsl)
add_subdirectory(exec)
add_subdirectory(op)

include(symbols.cmake)

pml_cc_library(
  NAME plaidml
  TYPE SHARED
  LINKOPTS
    # TODO: use platform-specific flags
  PROPS
    LINK_DEPENDS ${LINUX_LDS}
    LINK_DEPENDS ${MACOS_LD}
    LINK_DEPENDS ${WINDOWS_DEF}
    OUTPUT_NAME plaidml
    SOVERSION 1
  PRIVATE_DEPS
    plaidml::core::obj
    plaidml::edsl::obj
    plaidml::exec::obj
    plaidml::op::obj
    pmlc::rt::all_runtimes
    pmlc::target::all_obj
    pmlc::target::intel_gen
    pmlc::target::intel_gen_ocl_spirv
)

pml_cc_library(
  NAME testenv
  HDRS testenv.h
  SRCS testenv.cc
  DEPS
    plaidml::core
    plaidml::edsl
    plaidml::exec
    plaidml::op
    pmlc::rt::all_runtimes
    pmlc::target::all
    gflags
    gmock
    gtest
    half
)
