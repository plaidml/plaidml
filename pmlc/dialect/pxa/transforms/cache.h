// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "pmlc/dialect/pxa/ir/ops.h"

namespace pmlc::dialect::pxa {

LogicalResult cacheLoad(mlir::AffineParallelOp par, PxaLoadOp load);
LogicalResult cacheReduce(mlir::AffineParallelOp par, PxaReduceOp reduce);
LogicalResult cacheLoadAsVector(mlir::AffineParallelOp par, PxaLoadOp load);

} // namespace pmlc::dialect::pxa
