// Copyright 2020 Intel Corporation

#pragma once

#include "pmlc/dialect/pxa/ir/ops.h"

namespace pmlc::dialect::pxa {

// Tiles op in place, and returns the inner AffineParallelOp
mlir::AffineParallelOp performTiling(mlir::AffineParallelOp op,
                                     llvm::ArrayRef<int64_t> tileSizes);

// Undo tiling
mlir::AffineParallelOp undoTiling(mlir::AffineParallelOp op,
                                  llvm::ArrayRef<int64_t> tileSizes);

} // namespace pmlc::dialect::pxa
