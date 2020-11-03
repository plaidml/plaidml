pml_cc_test(
  NAME cc_test
  SRCS edsl_test.cc
  DEPS
    plaidml::testenv
    pmlc::testing
)

pml_lit_test(
  NAME lit
  DATA ::cc_test
)
