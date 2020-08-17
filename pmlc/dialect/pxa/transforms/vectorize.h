// Copyright 2020 Intel Corporation

#pragma once

#include "pmlc/dialect/pxa/ir/ops.h"

namespace pmlc::dialect::pxa {

// Explicitly attempt to vectorize on a specific loop/index/size
LogicalResult performVectorization(mlir::AffineParallelOp op,
                                   mlir::BlockArgument index,
                                   unsigned vectorSize);

// Attempt to vectorize a loop on the stride 1 index of it's output
LogicalResult simpleVectorize(mlir::AffineParallelOp op, unsigned vecSize);

} // namespace pmlc::dialect::pxa
