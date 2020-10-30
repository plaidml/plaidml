// Copyright 2020, Intel Corporation
#pragma once

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace pmlc::dialect::pxa {

/// Structure describing layout change in terms of affine map from previous
/// layout to new one. Additionally it holds shape and vectorization
/// of reordered memory as extracting this information from affine map
/// is not trivial.
struct ReorderDesc {
  // Map from original memory layout into target layout.
  mlir::AffineMap reorderMap;
  // Shape of memory after reordering to new layout.
  mlir::SmallVector<int64_t, 6> reorderedShape;
  // Shape of vector operations in new layout.
  mlir::SmallVector<int64_t, 6> reorderedVector;
};

/// Creates operations that copy `srcMem` into new memory with layout
/// transformed according to ReorderDesc.
/// Created operations consist of:
/// 1. `alloc` allocating output memory;
/// 2. `affine.parallel` over input memory shape;
/// 3. `pxa.load` reading input memory with identity map;
/// 4. `pxa.reduce` writing to output memory with transformed map.
/// Returns reordered memory.
mlir::Value createReorder(mlir::Location loc, mlir::OpBuilder &builder,
                          ReorderDesc &desc, mlir::Value srcMem);

/// Reorders memory `srcMem` to new layout and uses it in all operations
/// reading from original memory.
/// Returns reordered memory.
void replaceMemoryLayoutForReading(mlir::Value reorderedMemory,
                                   mlir::Value replaceMemory,
                                   ReorderDesc &desc);

/// Attempts to convert layout for `memory` as specified in ReorderDesc.
/// Returns failure if layout cannot be changed, for example if
/// memory is allocated outside of function or is used by unsupported
/// operations.
mlir::LogicalResult convertMemoryLayout(mlir::Value memory, ReorderDesc &desc);

} // namespace pmlc::dialect::pxa
