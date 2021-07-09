// Copyright 2021, Intel Corporation

#include "pmlc/dialect/pxa/ir/stride_range.h"

#include <algorithm>
#include <numeric>
#include <utility>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "pmlc/util/util.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

StrideRange::StrideRange(BlockArgument arg)
    : valid(false), minVal(0), maxVal(0), stride(0) {
  if (auto ap = dyn_cast<AffineParallelOp>(arg.getOwner()->getParentOp())) {
    auto rangeExpr = util::getRangesValueMap(ap).getResult(arg.getArgNumber());
    auto rangeConstantExpr = rangeExpr.dyn_cast<AffineConstantExpr>();
    if (!rangeConstantExpr) {
      return;
    }
    int64_t range = rangeConstantExpr.getValue();
    if (range < 1) {
      return;
    }
    auto steps = ap.getSteps();
    int64_t step = steps[arg.getArgNumber()];
    if (step <= 0) {
      return;
    }
    stride = 1;
    minVal = 0;
    // This is a correction to deal with the fact that strides are measured
    // relative to loop iterations not indexes.
    maxVal = (range - 1) / step;
    valid = true;
    if (minVal == maxVal) {
      stride = 0;
    }
  }
}

StrideRange &StrideRange::operator*=(int64_t factor) {
  minVal *= factor;
  maxVal *= factor;
  stride *= factor;
  if (factor < 0) {
    std::swap(minVal, maxVal);
  }
  return *this;
}

StrideRange &StrideRange::operator+=(const StrideRange &rhs) {
  valid = valid && rhs.valid;
  minVal += rhs.minVal;
  maxVal += rhs.maxVal;
  stride = std::gcd(stride, rhs.stride);
  return *this;
}

void StrideRange::unionEquals(const StrideRange &rhs) {
  valid = valid && rhs.valid;
  minVal = std::min(minVal, rhs.minVal);
  maxVal = std::max(maxVal, rhs.maxVal);
  stride = std::gcd(stride, rhs.stride);
}

std::ostream &operator<<(std::ostream &os, const StrideRange &val) {
  os << '(' << val.minVal << ", " << val.maxVal << "]:" << val.stride;
  return os;
}

} // namespace pmlc::dialect::pxa
