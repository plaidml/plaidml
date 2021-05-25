pml_py_library(
  NAME py
  SRC
    hypershere.py
    op.py
    torus.py
  DEPS
    plaidml::py
)

pml_py_test(
  NAME py_test
  SRC storti_integral_tests.py
  DEPS
    ::py
  CHECKS
      core
      smoke
      test
)
