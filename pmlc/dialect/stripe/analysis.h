// Copyright 2019, Intel Corporation

#pragma once

#include <algorithm>
#include <map>
#include <vector>

#include "pmlc/dialect/stripe/affine_poly.h"
#include "pmlc/dialect/stripe/mlir.h"
#include "pmlc/dialect/stripe/ops.h"

namespace pmlc {
namespace dialect {
namespace stripe {

// An affine range, which is a is basically the 'extent' of an affine
// polynomial, and also supports the operations of 'unioning'
struct AffineRange {
  int64_t min;      // Inclusive minimum value
  int64_t max;      // Inclusive maximum value
  uint64_t stride;  // Step size of the affine

  AffineRange() : min(0), max(0), stride(0) {}
  AffineRange(int64_t _min, int64_t _max, uint64_t _stride = 1);
  AffineRange& operator*=(int64_t x);
  AffineRange& operator+=(const AffineRange& x);
  AffineRange& operator|=(const AffineRange& x);
  explicit AffineRange(const AffinePolynomial& x);
  explicit AffineRange(Value* x) : AffineRange(AffinePolynomial(x)) {}
};

// Operator overload for affine ops as well as union
AFFINE_OP_OVERLOADS(AffineRange)
inline AffineRange operator|(const AffineRange& a, const AffineRange& b) {
  AffineRange r = a;
  r |= b;
  return r;
}

struct FlatTensorAccess {
  Value* base;
  TensorType base_type;
  std::vector<AffinePolynomial> access;

  // Comparisons
  bool operator<(const FlatTensorAccess& rhs) const;
  bool operator==(const FlatTensorAccess& rhs) const;
  CMP_OVERLOADS(FlatTensorAccess)
};

// For a tensor-reference, compute some information about the base allocation as
// well as its access polynomials
FlatTensorAccess ComputeAccess(Value* tensor);

// Check if the parallel-for contains a constraint as its final op, and also
// that any ops before the constraint are no-side-effect
bool SafeConstraintInterior(ParallelForOp op);

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
