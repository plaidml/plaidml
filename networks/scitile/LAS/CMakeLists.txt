pml_py_test(
  NAME cg_test
  SRC cg_tests.py
  DEPS
    plaidml::py
  CHECKS
    core
    smoke
    test
)
