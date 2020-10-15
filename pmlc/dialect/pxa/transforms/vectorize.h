// Copyright 2020 Intel Corporation

#pragma once

#include "pmlc/dialect/pxa/ir/ops.h"

namespace pmlc::dialect::pxa {

static constexpr const char *kVectorizeStrategy_Outputs = "outputs";
static constexpr const char *kVectorizeStrategy_Recursive = "recursive";
static constexpr const char *kVectorizeStrategy_Simple = "simple";

// Explicitly attempt to vectorize on a specific loop/index/size
mlir::LogicalResult performVectorization(mlir::AffineParallelOp op,
                                         mlir::BlockArgument index,
                                         unsigned vectorSize);

// Attempt to vectorize a loop on the stride 1 index of its output
mlir::LogicalResult vectorizeOverOutputs(mlir::AffineParallelOp op,
                                         unsigned vecSize);

// Attempt to vectorize a buffer (given it's allocation)
mlir::LogicalResult vectorizeBuffer(mlir::AllocOp op);

} // namespace pmlc::dialect::pxa
