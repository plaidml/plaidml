set(_SRCS ops.cc)
set(_HDRS ops.h)

pml_cc_library(
  NAME lib
  TYPE STATIC
  HDRS ${_HDRS}
  SRCS ${_SRCS}
  DEPS
    plaidml::core
    plaidml::edsl
    pmlc::util
    LLVMSupport
)

pml_cc_library(
  NAME obj
  TYPE OBJECT
  HDRS ${_HDRS}
  SRCS ${_SRCS}
  DEPS
    plaidml::core::obj
    plaidml::edsl::obj
    pmlc::util
    LLVMSupport
)
