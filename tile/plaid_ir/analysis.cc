// Copyright 2019, Intel Corporation

#include "tile/plaid_ir/analysis.h"

#include <boost/math/common_factor.hpp>

#include "tile/plaid_ir/ops.h"

namespace vertexai {
namespace tile {
namespace plaid_ir {

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

void AffineRange::merge(const AffineRange& x) {
  min = std::min(min, x.min);
  max = std::max(max, x.max);
  stride = boost::math::gcd(stride, x.stride);
}

AffineRange UnboundedRange(Value* aff) {
  if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(aff)) {
    // Extract the parallel for this affine is an argument of
    auto pf = mlir::cast<ParallelForOp>(ba->getOwner()->getContainingOp());
    // Extract the appropriate range attribute
    Attribute r = pf.ranges().getValue()[ba->getArgNumber()];
    // Return a new range
    return AffineRange(0, r.cast<IntegerAttr>().getInt() - 1);
  }
  auto base = aff->getDefiningOp();
  if (auto op = mlir::dyn_cast<AffineConstOp>(base)) {
    int64_t value = op.value().getSExtValue();
    return AffineRange(value, value);
  } else if (auto op = mlir::dyn_cast<AffineMulOp>(base)) {
    AffineRange r = UnboundedRange(op.input());
    r *= op.scale().getSExtValue();
    return r;
  } else if (auto op = mlir::dyn_cast<AffineAddOp>(base)) {
    AffineRange out;
    for (Value* v : op.inputs()) {
      out += UnboundedRange(v);
    }
    return out;
  } else {
    throw std::runtime_error("Invalid affine in UnboundedRange");
  }
}

}  // namespace plaid_ir
}  // namespace tile
}  // namespace vertexai
