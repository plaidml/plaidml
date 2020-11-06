pml_cc_library(
  NAME bilp
  HDRS
    ilp_solver.h
    tableau.h
  SRCS
    ilp_solver.cc
    tableau.cc
  DEPS
    pmlc::util
    pmlc::util::math
    Boost::boost
    gtest
)
