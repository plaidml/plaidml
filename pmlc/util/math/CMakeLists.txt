pml_cc_library(
  NAME math
  HDRS
    basis.h
    bignum.h
    matrix.h
    polynomial.h
    util.h
  SRCS
    basis.cc
    bignum.cc
    matrix.cc
    polynomial.cc
    util.cc
  DEPS
    pmlc::util
    Boost::boost
)
