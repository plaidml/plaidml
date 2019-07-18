// Copyright 2019, Intel Corporation

#include "tile/plaid_ir/analysis.h"

#include <boost/math/common_factor.hpp>

#include "tile/plaid_ir/ops.h"

namespace vertexai {
namespace tile {
namespace plaid_ir {

AffinePolynomial::AffinePolynomial() : constant(0) {}

AffinePolynomial::AffinePolynomial(int64_t x) : constant(x) {}

AffinePolynomial::AffinePolynomial(Value* x) : constant(0) {
  if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(x)) {
    // This is a basic index: done
    terms.emplace(ba, 1);
    return;
  }
  auto dop = x->getDefiningOp();
  if (auto op = mlir::dyn_cast<AffineConstOp>(dop)) {
    // This is a constant
    constant = op.value().getSExtValue();
  } else if (auto op = mlir::dyn_cast<AffineMulOp>(dop)) {
    *this = AffinePolynomial(op.input());
    *this *= op.scale().getSExtValue();
  } else if (auto op = mlir::dyn_cast<AffineAddOp>(dop)) {
    for (Value* v : op.inputs()) {
      *this += AffinePolynomial(v);
    }
  } else {
    throw std::runtime_error("Invalid affine in ComputeAffineRange");
  }
}

AffinePolynomial& AffinePolynomial::operator*=(int64_t x) {
  constant *= x;
  if (x == 0) {
    terms.clear();
  } else {
    for (auto& kvp : terms) {
      kvp.second *= x;
    }
  }
  return *this;
}

AffinePolynomial& AffinePolynomial::operator+=(const AffinePolynomial& x) {
  constant += x.constant;
  for (const auto& kvp : x.terms) {
    terms[kvp.first] += kvp.second;
    if (terms[kvp.first] == 0) {
      terms.erase(kvp.first);
    }
  }
  return *this;
}

AffineRange::AffineRange(int64_t _min, int64_t _max, uint64_t _stride) : min(_min), max(_max), stride(_stride) {
  if (min == max) {
    stride = 0;
  }
}

AffineRange& AffineRange::operator*=(int64_t x) {
  min *= x;
  max *= x;
  if (x < 0) {
    std::swap(min, max);
  }
  stride *= x;
  return *this;
}

AffineRange& AffineRange::operator+=(const AffineRange& x) {
  min += x.min;
  max += x.max;
  stride = boost::math::gcd(stride, x.stride);
  return *this;
}

AffineRange& AffineRange::operator|=(const AffineRange& x) {
  min = std::min(min, x.min);
  max = std::max(max, x.max);
  stride = boost::math::gcd(stride, x.stride);
  return *this;
}

AffineRange::AffineRange(const AffinePolynomial& poly) : min(poly.constant), max(poly.constant), stride(0) {
  // Go over each term and adjust
  for (const auto& kvp : poly.terms) {
    // Update the stride
    stride = boost::math::gcd(stride, uint64_t(std::abs(kvp.second)));
    // Extract the parallel for this affine is an argument of
    auto pf = mlir::cast<ParallelForOp>(kvp.first->getOwner()->getContainingOp());
    // Extract the appropriate attribute from ranges
    Attribute ra = pf.ranges().getValue()[kvp.first->getArgNumber()];
    // Turn the range into an integer
    int64_t range = ra.cast<IntegerAttr>().getInt();
    // Update min/max
    if (kvp.second >= 0) {
      max += (range - 1) * kvp.second;
    } else {
      min += (range - 1) * kvp.second;
    }
  }
}

}  // namespace plaid_ir
}  // namespace tile
}  // namespace vertexai
