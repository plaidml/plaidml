// Copyright 2020 Intel Corporation
#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace pmlc::dialect::pxa {

/// Tiles affine parallel loop for GPU up to "maxThreads" inner tile size.
void gpuThreadParallelOp(unsigned maxThreads, mlir::AffineParallelOp op);

} // namespace pmlc::dialect::pxa
