// Copyright 2020 Intel Corporation

#pragma once

#include "pmlc/dialect/pxa/ir/ops.h"

namespace pmlc::dialect::pxa {

LogicalResult performVectorization(mlir::AffineParallelOp op,
                                   mlir::BlockArgument index,
                                   unsigned vectorSize);

} // namespace pmlc::dialect::pxa
