// Copyright 2020 Intel Corporation

#pragma once

#include "llvm/ADT/SmallVector.h"

#include "pmlc/util/enums.h"

namespace mlir {
class Operation;
}

namespace pmlc::dialect::tile {

struct PaddingInfo {
  util::AggregationKind agg;
  llvm::SmallVector<int64_t, 4> lower;
  llvm::SmallVector<int64_t, 4> upper;
};

llvm::Optional<PaddingInfo> getPaddingInfo(mlir::Operation *op);

} // namespace pmlc::dialect::tile
