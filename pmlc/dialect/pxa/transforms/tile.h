// Copyright 2019, Intel Corporation

#pragma once

#include "pmlc/dialect/pxa/ir/ops.h"

namespace pmlc::dialect::pxa {

void Tile(AffineParallelOp op, llvm::ArrayRef<int64_t> tileSizes);

}  // namespace pmlc::dialect::pxa
