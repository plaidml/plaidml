pml_py_test(
  NAME storti_py_test
  SRC storti_integral_tests.py
  DEPS
    plaidml::py
  CHECKS
      core
      smoke
      test
)
