// Copyright 2019, Intel Corporation

#pragma once

#include <algorithm>
#include <map>
#include <vector>

#include "pmlc/dialect/stripe/mlir.h"

namespace pmlc {
namespace dialect {
namespace stripe {

// A macro to add operator overloads for a class
#define AFFINE_OP_OVERLOADS(X)              \
  inline X operator*(X lhs, int64_t rhs) {  \
    lhs *= rhs;                             \
    return lhs;                             \
  }                                         \
  inline X operator*(int64_t lhs, X rhs) {  \
    rhs *= lhs;                             \
    return rhs;                             \
  }                                         \
  inline X operator+(X lhs, const X& rhs) { \
    lhs += rhs;                             \
    return lhs;                             \
  }                                         \
  inline X operator-(X lhs, const X& rhs) { \
    lhs += (-1 * rhs);                      \
    return lhs;                             \
  }

// A macro to secondary comparisons for a class
#define CMP_OVERLOADS(X)                                           \
  inline bool operator!=(const X& rhs) { return !(*this == rhs); } \
  inline bool operator>(const X& rhs) { return rhs < *this; }      \
  inline bool operator<=(const X& rhs) { return !(rhs < *this); }  \
  inline bool operator>=(const X& rhs) { return !(*this < rhs); }

// An affine 'polynomial', basically a series of terms, each consisting of a
// multiplier and an index (a Affine that is also a BlockArgument), as well as a
// constant offset.  We can 'flatten' any affine expression into such a
// polynomial.
struct AffinePolynomial {
  std::map<mlir::BlockArgument*, int64_t> terms;
  int64_t constant;
  // Make an empty polynomial
  AffinePolynomial();
  // Make a constant polynomial
  explicit AffinePolynomial(int64_t x);
  // Make a polynomial from an affine expression
  explicit AffinePolynomial(Value* x);
  // Perform operations on a polynomial
  AffinePolynomial& operator*=(int64_t x);
  AffinePolynomial& operator+=(const AffinePolynomial& x);
  // Comparisons
  bool operator<(const AffinePolynomial& rhs) const;
  bool operator==(const AffinePolynomial& rhs) const;
  CMP_OVERLOADS(AffinePolynomial)
};

AFFINE_OP_OVERLOADS(AffinePolynomial)

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
