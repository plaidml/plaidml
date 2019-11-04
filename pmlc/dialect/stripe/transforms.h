// Copyright 2019, Intel Corporation

#pragma once

#include "pmlc/dialect/stripe/mlir.h"
#include "pmlc/dialect/stripe/ops.h"

namespace pmlc {
namespace dialect {
namespace stripe {

void Tile(ParallelForOp op, llvm::ArrayRef<int64_t> tile_sizes);
void ExtractConstraintCase(ParallelForOp op, bool ge);
void LimitLower(ParallelForOp op, size_t arg, int64_t val);
void LimitUpper(ParallelForOp op, size_t arg, int64_t val);
void LiftConstraint(ParallelForOp op);
bool SplitFor(ParallelForOp op);

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
