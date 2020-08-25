// Copyright 2020 Intel Corporation

#pragma once

#include "pmlc/dialect/pxa/ir/ops.h"

void normalizeAffineParallel(mlir::AffineParallelOp op);

namespace pmlc::dialect::pxa {

// Promotes the loop body of an affine.parallel to its containing block if no
// induction variables are present.
void promoteIfEmptyIVs(mlir::AffineParallelOp op);

void elideSingleIterationIndexes(mlir::AffineParallelOp op);

} // namespace pmlc::dialect::pxa
