pml_cc_library(
  NAME testing
  SRCS gtest_main.cc
  DEPS
    pmlc::util
    gflags
    gtest
)
