add_subdirectory(ops)

set(_SRCS plaidml_program_builder.cpp)
set(_HDRS plaidml_program_builder.hpp)

pml_cc_library(
  NAME program_builder
  TYPE STATIC
  HDRS ${_HDRS}
  SRCS ${_SRCS}
  # TODO: Confirm & clean DEPS
  DEPS
    plaidml::core
    IE::inference_engine
    # pmlc::ast
    # pmlc::util
    # LLVMSupport
)
