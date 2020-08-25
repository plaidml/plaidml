// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/ir/ops.h"

namespace pmlc::dialect::pxa {

struct CacheInfo {
  mlir::AffineParallelOp copyLoopOp;
  RelativeAccessPattern relativeAccess;
};

mlir::Optional<CacheInfo> cacheLoad(mlir::AffineParallelOp par, PxaLoadOp load);

mlir::Optional<CacheInfo> cacheReduce(mlir::AffineParallelOp par,
                                      PxaReduceOp reduce);

LogicalResult cacheLoadAsVector(mlir::AffineParallelOp par, PxaLoadOp load,
                                int64_t reqVecSize = 0);

} // namespace pmlc::dialect::pxa
