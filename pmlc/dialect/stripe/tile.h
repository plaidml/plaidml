// Copyright 2019, Intel Corporation

#pragma once

#include "pmlc/dialect/stripe/mlir.h"
#include "pmlc/dialect/stripe/ops.h"

namespace pmlc {
namespace dialect {
namespace stripe {

void Tile(ParallelForOp op, llvm::ArrayRef<int64_t> tile_sizes);

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
