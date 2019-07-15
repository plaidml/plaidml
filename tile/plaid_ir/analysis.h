// Copyright 2019, Intel Corporation

#pragma once

#include <algorithm>

#include "tile/plaid_ir/mlir.h"

namespace vertexai {
namespace tile {
namespace plaid_ir {

struct AffineRange {
  AffineRange() : min(0), max(0), stride(0) {}
  AffineRange(int64_t _min, int64_t _max, uint64_t _stride = 1);
  AffineRange& operator*=(int64_t x);
  AffineRange& operator+=(const AffineRange& x);
  void merge(const AffineRange& x);
  int64_t min;      // Inclusive minimum value
  int64_t max;      // Inclusive maximum value
  uint64_t stride;  // Step size of the affine
};

inline AffineRange operator*(int64_t x, const AffineRange& aff) {
  AffineRange r = aff;
  r *= x;
  return r;
}
inline AffineRange operator*(const AffineRange& aff, int64_t x) {
  AffineRange r = aff;
  r *= x;
  return r;
}
inline AffineRange operator+(const AffineRange& lhs, const AffineRange& rhs) {
  AffineRange r = lhs;
  r += rhs;
  return r;
}

AffineRange UnboundedRange(Value* aff);

}  // namespace plaid_ir
}  // namespace tile
}  // namespace vertexai
