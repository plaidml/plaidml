pml_cc_test(
  NAME examples_cc_test
  SRCS 
    complex_conv_2d.cc
    conv_1d.cc
    conv_2d_dilated.cc
    gemm.cc
    gemv.cc
    max_pool_1d.cc
    quantize.cc
  DEPS
    plaidml::testenv
    pmlc::testing
  CHECKS
    core
    smoke
    test
)

pml_py_test(
  NAME examples_py_test
  SRC example_tests.py
  CHECKS
    core
    smoke
    test
)

