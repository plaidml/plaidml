// Copyright 2020, Intel Corporation
#pragma once

#include <functional>

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "pmlc/dialect/pxa/ir/interfaces.h"
#include "pmlc/dialect/pxa/transforms/layout_utils.h"

namespace pmlc::dialect::pxa {

/// Structure holding information about single read operation.
struct MemoryReadDesc {
  // Operation this structure describes.
  PxaReadOpInterface readOp;
  // Affine map of read operation.
  mlir::AffineMap readMap;
  // Vectorization of read operation.
  mlir::SmallVector<int64_t, 4> readVector;
  // Constraints for dimensions of `readMap`.
  mlir::FlatAffineConstraints dimensionConstraints;
  // Iteration order for input dimensions in `readMap`
  // (from least to most frequent).
  mlir::SmallVector<unsigned, 6> iterationOrder;
};

/// Structure holding information about single write operation.
struct MemoryWriteDesc {
  // Vectorization of write operation.
  mlir::SmallVector<int64_t, 4> writeVector;
};

/// Structure describing memory and its usage.
struct MemoryUsageDesc {
  // IR value representing memory.
  mlir::Value value;
  // Shape of memory.
  mlir::SmallVector<int64_t, 4> shape;
  // Number of elements in memory.
  int64_t count;
  // List of descriptions of reads accessing memory.
  mlir::SmallVector<MemoryReadDesc, 1> reads;
  // List of descriptions of writes accessing memory.
  mlir::SmallVector<MemoryWriteDesc, 1> writes;
};

/// OperandSchedule represents some schedule cost of each operand
/// of `affine.parallel`. Lower cost signifies that such dimension
/// should be prioritezed for layout optimization, for example
/// it changes more frequently.
using OperandSchedule = mlir::SmallVector<int64_t, 4>;
/// LoopNestSchedule represents operands cost of `affine.parallel` loop nest
/// ordered from outer to inner-most loop.
using LoopNestSchedule = mlir::SmallVector<OperandSchedule, 6>;
/// Schedule model represents a function that assigns cost to
/// `affine.parallel` loop nest.
/// Argument "loopNest" is a sequence of parallel loops from outer
/// to inner-most, same as expected returned order.
/// Result should be of same size as "loopNest", and each
/// element of result should have same size as number of dimensions
/// in corresponding input loop.
using ScheduleModel = std::function<LoopNestSchedule(
    mlir::ArrayRef<mlir::AffineParallelOp> /*loopNest*/)>;

/// Walks over all affine read and write operations in "func" and gathers
/// information about global memory and its accesses.
/// Only memory that is allocated outside of any `affine.parallel` is considered
/// global. ScheduleModel argument is used to assign "iterationOrder" to
/// each memory access.
/// Returns map from indirectly defined memory to its description.
mlir::DenseMap<mlir::Value, MemoryUsageDesc>
gatherGlobalMemoryDescs(mlir::FuncOp func, const ScheduleModel &model);

/// Function complying to ScheduleModel interface, that naively assigns
/// lower cost to more inner loops and later operands.
/// For example loop nest:
/// ```mlir
/// affine.parallel (%i, %j)
///   affine.parallel (%k, %l, %m)
/// ```
/// will return following schedule:
/// ```c
/// {{5, 4},
///  {3, 2, 1}}
/// ```
LoopNestSchedule
naiveScheduleModel(mlir::ArrayRef<mlir::AffineParallelOp> loopNest);

/// Generates reorder description from original memory layout to one
/// that optimizes memory reads.
/// Returns llvm::None when more optimal layout cannot be found.
///
/// New layout is selected by performing two transformations:
/// 1. Expanding number of dimensions to try to reduce number of loop variables
///    each dimension depends on. Additionally vectorized dimensions
///    are separated as non-empty dimensions not depending on any loop variable.
/// 2. Permuting/sorting separated dimensions in order of loops whose variables
///    are used in each dimension.
mlir::Optional<ReorderDesc> optimizeLayoutForReads(MemoryUsageDesc &desc);

/// Type of function that is expected to create reordering operation
/// from "srcMemory" to layout described by "reorderDesc".
/// See: `createReorder` in `pmlc/dialect/pxa/transforms/layout_utils.h`.
using ReorderCreator = std::function<mlir::Value(
    mlir::Location /*location*/, mlir::OpBuilder & /*builder*/,
    ReorderDesc & /*reorderDesc*/, mlir::Value /*srcMemory*/)>;

/// Helper function that reorders memory for all reads to layout specified
/// by "reorderDesc". Reorder is created using "creator" and all memory usages
/// in read operations are replaced to newly reordered memory.
/// It guards against reordering same memory used by two read operations
/// twice, which could happen when directly using "creator".
void reorderMemoryReads(const ReorderCreator &creator, ReorderDesc &reorderDesc,
                        MemoryUsageDesc &memoryDesc);

// ============================================================================
// Helper affine map transformations
// ============================================================================

/// Expand affine map dimensions based on integral constraints and vector shape.
/// Aim of this transformation is to separate vector dimension and reduce
/// number of input dimensions each result dimension depends on.
///
/// Input:
///   map (A)     = (d0, d1) -> (d0 + d1, d0 * 16 + d1 * 8)
///   shape       = <7, 96>
///   vector      = <1, 8>
///   constraints = {0 <= d1 < 2, 0 <= d0 < 6}
/// Output:
///   reorder map (B)  = (d0, d1) -> (d0, d1 floordiv 8 floordiv 2,
///                                   d1 floordiv 8 % 2, 0)
///   reordered shape  = <7, 6, 2, 8>
///   reordered vector = <1, 1, 1, 8>
///
/// Note: to obtain affine map from input space to expanded space composition
///       A o B can be used (with simplification).
/// A o B = (d0, d1) -> (d0 + d1, d0, d1, 0)
ReorderDesc expandAffineMap(mlir::AffineMap map, mlir::ArrayRef<int64_t> vector,
                            mlir::FlatAffineConstraints &constraints);

/// Create affine permutation map that sorts resulting space dimensions in order
/// of increasing schedule.
/// Vectorized dimensions are alway put last.
/// In basic case dimension latest in schedule and used in expression determines
/// the order.
/// If two dimensions have same input dimension as appearing latest in schedule,
/// remaining dimensions specify their order.
/// If two output dimensions use exactly the same input dimensions in their
/// expressions, original order is preserved (stable sort).
///
/// Input:
///   map (A)    = (d0, d1) -> (d0 + d1, d0, d1, 0)
///   shape      = <7, 6, 2, 8>
///   vector     = <1, 1, 1, 8>
///   schedule   = <1, 0>
/// Output:
///   reorder map (B)  = (d0, d1, d2, d3) -> (d2, d0, d1, d3)
///   reordered shape  = <2, 7, 6, 8>
///   reordered vector = <1, 1, 1, 8>
///
/// Note: to obtain affine map from input space to sorted space composition
///       A o B can be used.
///   A o B = (d0, d1) -> (d1, d0 + d1, d0, 0)
mlir::AffineMap sortAffineMap(mlir::AffineMap map,
                              mlir::ArrayRef<int64_t> vector,
                              mlir::ArrayRef<unsigned> schedule);

/// Tile affine map using integral constraints to optimize specified schedule.
/// Returns llvm::None if current affine map is already optimal.
/// In essence this function first performs expansion of dimensions, then
/// sorts them according to schedule.
///
/// Input:
///   map (A)     = (d0, d1) -> (d0 + d1, d0 * 16 + d1 * 8)
///   shape       = <7, 96>
///   vector      = <1, 8>
///   constraints = {0 <= d1 < 2, 0 <= d0 < 6}
///   schedule    = <1, 0>
/// Output:
///   reorder map (B)  = (d0, d1) -> (d1 floordiv 8 % 2, d0,
///                                   d1 floordiv 8 floordiv 2, 0)
///   reordered shape  = <2, 7, 6, 8>
///   reordered vector = <1, 1, 1, 8>
///
/// Note: to obtain affine map from input space to tiled space composition
///       A o B can be used (with simplification).
///   A o B = (d0, d1) -> (d1, d0 + d1, d0, 0)
mlir::Optional<ReorderDesc>
tileAffineMap(mlir::AffineMap map, mlir::ArrayRef<int64_t> shape,
              mlir::ArrayRef<int64_t> vector,
              mlir::FlatAffineConstraints constraints,
              mlir::ArrayRef<unsigned> schedule);

} // namespace pmlc::dialect::pxa
