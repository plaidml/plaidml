// Copyright 2021, Intel Corporation

#pragma once

#include "mlir/IR/Value.h"

namespace pmlc::dialect::pxa {

struct StrideRange {
  bool valid;
  int64_t minVal;
  int64_t maxVal;
  int64_t stride;

  explicit StrideRange(int64_t val)
      : valid(true), minVal(val), maxVal(val), stride(0) {}

  explicit StrideRange(int64_t min, int64_t max, int64_t stride)
      : valid(true), minVal(min), maxVal(max), stride(stride) {
    if (min == max) {
      stride = 0;
    }
  }

  explicit StrideRange(mlir::BlockArgument arg);

  StrideRange &operator*=(int64_t factor);
  StrideRange operator*(int64_t factor) const {
    StrideRange ret = *this;
    ret *= factor;
    return ret;
  }

  StrideRange &operator+=(const StrideRange &rhs);
  StrideRange operator+(const StrideRange &rhs) const {
    StrideRange ret = *this;
    ret += rhs;
    return ret;
  }

  int64_t count() const {
    if (!valid) {
      return 0;
    }
    if (stride == 0) {
      return 1;
    }
    return (maxVal - minVal) / stride + 1;
  }

  void unionEquals(const StrideRange &rhs);
};

std::ostream &operator<<(std::ostream &os, const StrideRange &val);

} // namespace pmlc::dialect::pxa
