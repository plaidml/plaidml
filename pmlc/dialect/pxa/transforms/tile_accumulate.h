// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "pmlc/dialect/pxa/ir/ops.h"

namespace pmlc::dialect::pxa {

AffineParallelOp tileAccumulations(AffineParallelOp op,
                                   bool skipTrivial = true);

} // namespace pmlc::dialect::pxa
