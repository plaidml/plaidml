// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace pmlc::dialect::tile {

// Flatten src to a linear tensor. Must set the proper builder insertion point
// before this function. Return the linear tensor.
mlir::Value flattenTensor(mlir::OpBuilder &builder, mlir::Value src);

// Reshape src according to dstShape. Must set the proper builder insertion
// point before this function. Return null Value if src indices and dst indices
// are out-of-order.
mlir::Value reshapeTensor(mlir::OpBuilder &builder, mlir::Value src,
                          llvm::ArrayRef<int64_t> dstShape);

} // namespace pmlc::dialect::tile
