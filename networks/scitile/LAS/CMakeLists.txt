pml_py_test(
  NAME cg_test
  SRC cg_tests.py
  DEPS
    plaidml::plaidml
  CHECKS
    core
    smoke
    test
)
