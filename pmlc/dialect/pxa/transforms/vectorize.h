// Copyright 2020 Intel Corporation

#pragma once

#include "pmlc/dialect/pxa/ir/ops.h"

namespace pmlc::dialect::pxa {

bool performVectorization(mlir::AffineParallelOp op, mlir::BlockArgument index,
                          unsigned vectorSize, unsigned minElemWidth);

} // namespace pmlc::dialect::pxa
