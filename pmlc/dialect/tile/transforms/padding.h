// Copyright 2020 Intel Corporation

#pragma once

#include <algorithm>

#include "llvm/ADT/SmallVector.h"

#include "pmlc/util/enums.h"

namespace mlir {
class Operation;
class AffineMap;
} // namespace mlir

namespace pmlc::dialect::tile {

struct BoundRange {
  int64_t min;
  int64_t max;

  explicit BoundRange(int64_t val) : min(val), max(val) {}
  BoundRange(int64_t min, int64_t max) : min(min), max(max) {}

  BoundRange operator+(const BoundRange &rhs) const;
  BoundRange operator*(const BoundRange &rhs) const;
};

llvm::SmallVector<BoundRange, 4> computePaddingBounds(mlir::AffineMap access,
                                                      mlir::AffineMap lower,
                                                      mlir::AffineMap upper);

struct PaddingInfo {
  util::AggregationKind agg;
  llvm::SmallVector<int64_t, 4> lower;
  llvm::SmallVector<int64_t, 4> upper;
};

llvm::Optional<PaddingInfo> getPaddingInfo(mlir::Operation *op);

} // namespace pmlc::dialect::tile
