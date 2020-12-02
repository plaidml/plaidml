add_subdirectory(bridge)
add_subdirectory(core)
add_subdirectory(edsl)
add_subdirectory(exec)
add_subdirectory(op)

include(symbols.cmake)

if (NOT WIN32)
  pml_cc_library(
    NAME plaidml
    TYPE SHARED
    LINKOPTS
      # TODO: use platform-specific flags
      -Wl,--version-script ${LINUX_LDS}
    PROPS
      LINK_DEPENDS ${LINUX_LDS}
      LINK_DEPENDS ${MACOS_LD}
      OUTPUT_NAME plaidml
      SOVERSION 1
    PRIVATE_DEPS
      plaidml::core::obj
      plaidml::edsl::obj
      plaidml::exec::obj
      plaidml::op::obj
      pmlc::rt::all_runtimes
      pmlc::target::all_obj
  )
else()
# For Windows build both static and shared lib
  pml_cc_library(
    NAME plaidml
    TYPE STATIC
    LINKOPTS
    PROPS
      OUTPUT_NAME plaidml
      SOVERSION 1
    DEPS
      plaidml::core::obj
      plaidml::edsl::obj
      plaidml::exec::obj
      plaidml::op::obj
      pmlc::rt::all_runtimes
      pmlc::target::all_obj
  )

  pml_cc_library(
    NAME plaidml_dll
    TYPE SHARED
    LINKOPTS
    LINK_DEPENDS ${WINDOWS_DEF}
    PROPS
      OUTPUT_NAME plaidml
      SOVERSION 1
    DEPS
      plaidml::core::obj
      plaidml::edsl::obj
      plaidml::exec::obj
      plaidml::op::obj
      pmlc::rt::all_runtimes
      pmlc::target::all_obj
  )
endif()

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
