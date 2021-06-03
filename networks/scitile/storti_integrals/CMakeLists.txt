pml_py_library(
  NAME py
  SRC
    hypersphere.py
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

# the rules below are used to generate graphs for torus and hypersphere experiments. 
# these are disables because they take a while to generate and are only useful when the graphs need to be analyzed. 

# pml_py_test(
#   NAME torus
#   SRC torus.py
#   DEPS
#     plaidml::py
# )

# pml_py_test(
#   NAME hypersphere
#   SRC hypersphere.py
#   DEPS
#     plaidml::py
# )
