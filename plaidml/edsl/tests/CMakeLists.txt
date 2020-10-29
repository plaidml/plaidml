pml_cc_binary(
  NAME cc_test
  TYPE OBJECT
  SRCS edsl_test.cc
  DEPS
    plaidml::testenv
    pmlc::testing
)
