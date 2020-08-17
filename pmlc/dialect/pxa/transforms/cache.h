// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "pmlc/dialect/pxa/ir/ops.h"

namespace pmlc::dialect::pxa {

LogicalResult cacheLoad(mlir::AffineParallelOp par, mlir::AffineLoadOp load);
LogicalResult cacheReduce(mlir::AffineParallelOp par, PxaReduceOp reduce);

} // namespace pmlc::dialect::pxa
