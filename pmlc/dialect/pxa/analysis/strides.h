// Copyright 2020, Intel Corporation

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"

#include "pmlc/dialect/pxa/analysis/affine_expr.h"
#include "pmlc/dialect/pxa/ir/ops.h"

namespace pmlc::dialect::pxa {

// Get the step for a block argument as an IV of an affine.for or
// affine.parallel
int64_t getIVStep(mlir::BlockArgument arg);

struct StrideRange {
  bool valid;
  int64_t minVal;
  int64_t maxVal;
  int64_t stride;

  explicit StrideRange(int64_t val)
      : valid(true), minVal(val), maxVal(val), stride(0) {}

  explicit StrideRange(int64_t min, int64_t max, int64_t stride)
      : valid(true), minVal(min), maxVal(max), stride(stride) {
    if (min == max) {
      stride = 0;
    }
  }

  explicit StrideRange(mlir::BlockArgument arg);

  StrideRange &operator*=(int64_t factor);
  StrideRange operator*(int64_t factor) const {
    StrideRange ret = *this;
    ret *= factor;
    return ret;
  }

  StrideRange &operator+=(const StrideRange &rhs);
  StrideRange operator+(const StrideRange &rhs) const {
    StrideRange ret = *this;
    ret += rhs;
    return ret;
  }

  int64_t count() const {
    if (!valid) {
      return 0;
    }
    if (stride == 0) {
      return 1;
    }
    return (maxVal - minVal) / stride + 1;
  }

  void unionEquals(const StrideRange &rhs);
};

enum class BoundaryRegion {
  Interior,
  Exterior,
};

// A boundary function takes a BlockArgument and decides whether it is
// 'interior' or 'exterior'. The definition of 'interior' and 'exterior' are up
// to the caller, and is useful for computing access relative to some
// user-defined boundary.
//
// For example, this boundary could be defined as whether the owner (Block) of
// an argument is an ancestor of another reference block.
using BlockArgumentBoundaryFn =
    std::function<BoundaryRegion(mlir::BlockArgument arg)>;

// StrideInfo provides a simple 'stride' multiplier for each affine induction
// variable (from an affine.for or affine.parallel).  Basically, each step of
// the loop moves a pure affine expression by a fixed distance, 'strides' holds
// that distance.  Additionally it holds a fixed offset.
struct StrideInfo {
  int64_t offset;
  mlir::DenseMap<mlir::BlockArgument, int64_t> strides;

  explicit StrideInfo(mlir::BlockArgument arg)
      : offset(0), strides({{arg, 1}}) {}
  explicit StrideInfo(int64_t offset = 0) : offset(offset) {}

  bool operator==(const StrideInfo &rhs) const {
    return offset == rhs.offset && strides == rhs.strides;
  }

  bool operator!=(const StrideInfo &rhs) const { return !(*this == rhs); }

  StrideInfo &operator*=(int64_t factor);
  StrideInfo operator*(int64_t factor) const {
    StrideInfo ret = *this;
    ret *= factor;
    return ret;
  }

  StrideInfo &operator+=(const StrideInfo &rhs);
  StrideInfo operator+(const StrideInfo &rhs) const {
    StrideInfo r = *this;
    r += rhs;
    return r;
  }

  // Compute the outer and inner portion of stride info with respect to a given
  // block.
  StrideInfo outer(mlir::Block *block);
  StrideInfo inner(mlir::Block *block);

  // Compute the outer and inner portion of stride info with respect to a given
  // boundary function.
  StrideInfo outer(BlockArgumentBoundaryFn fn);
  StrideInfo inner(BlockArgumentBoundaryFn fn);

  // Return the range of a given stride info if it's computable
  StrideRange range() const;

  // Convert a StrideInfo back into an affine expression
  mlir::AffineValueExpr toValueExpr(mlir::MLIRContext *ctx) const;

  void print(mlir::raw_ostream &os, mlir::Block *relative = nullptr) const;
};

std::ostream &operator<<(std::ostream &os, const StrideInfo &x);

// Convert a vector of StrideInfo's into a value map
mlir::AffineValueMap convertToValueMap(mlir::MLIRContext *ctx,
                                       mlir::ArrayRef<StrideInfo> dims);

// Compute stride info for a given affine value (such an an induction variable
// or the result of an affine.apply). Return None if the expression is not a
// pure affine expression or if any of the gathered strides would be symbolic
mlir::Optional<StrideInfo> computeStrideInfo(mlir::Value expr);

// Compute stride info but for an affine expression over some set of values
mlir::Optional<StrideInfo> computeStrideInfo(mlir::AffineExpr expr,
                                             mlir::ValueRange args);

// Compute 'dimensionalized' strides for a given affine map and arguments
mlir::Optional<mlir::SmallVector<StrideInfo, 4>>
computeStrideInfo(mlir::AffineMap map, mlir::ValueRange args);

// Compute stride info as additionaly applied to a memRef.
mlir::Optional<StrideInfo> computeStrideInfo(mlir::MemRefType memRef,
                                             mlir::AffineMap map,
                                             mlir::ValueRange values);

// Helper that works on a affine load / store, etc.
mlir::Optional<StrideInfo> computeStrideInfo(pmlc::dialect::pxa::PxaLoadOp op);

mlir::Optional<StrideInfo>
computeStrideInfo(pmlc::dialect::pxa::PxaReduceOp op);

mlir::Optional<StrideInfo>
computeStrideInfo(pmlc::dialect::pxa::PxaVectorLoadOp op);

mlir::Optional<StrideInfo>
computeStrideInfo(pmlc::dialect::pxa::PxaVectorReduceOp op);

// For a given block + memory access:
// 1) What is the effect of all block args outside and including the
// block on the memory access, in terms of strides.
// 2) For all interior blocks/load width/etc, what is the range of elements
// accessed.
struct RelativeAccessPattern {
  explicit RelativeAccessPattern(mlir::Value memRef) : memRef(memRef) {}

  // The memref type of the access.
  mlir::Value memRef;

  // For each dimension of the access: the strides relative to all in-scope
  // block arguments.
  mlir::SmallVector<StrideInfo, 4> outer;

  // For each dimension of the access: the offset inside the block.
  mlir::SmallVector<StrideInfo, 4> inner;

  // For each dimension of the access: the StrideRange of the interior.
  mlir::SmallVector<StrideRange, 4> innerRanges;

  // For each dimension what is the number of accesses
  mlir::SmallVector<int64_t, 4> innerCount;

  // For each dimension what is the number of accesses including skipped elements
  mlir::SmallVector<int64_t, 4> wholeInnerCount;

  // For each dimension what is the minimal stride of the access.  Note:
  // dimensions with a count of 1 have a stride of 1 automatically
  mlir::SmallVector<int64_t, 4> innerStride() const;

  // Return the outer linearized strides relative to each block argument.
  mlir::Optional<StrideInfo> flatOuter() const;

  // Return the inner linearized strides relative to each block argument.
  mlir::Optional<StrideInfo> flatInner() const;

  mlir::MemRefType getMemRefType() const;

  // Return the total element count for all inner accesses.
  int64_t totalInnerCount() const;

  // Return the total bytes for all inner accesses.
  int64_t totalInnerBytes() const;

  // Merge another RelativeAccesPattern together by using a union.
  mlir::LogicalResult unionMerge(const RelativeAccessPattern &rhs);

  // Given a full set of outer indexes (in case some of them are unused in the
  // various stride-infos), return true if two distinct outer loop interations
  // can access the same memory element of the tensor.  i.e. do any outer loops
  // interations ever alias with other outer loop iterations.
  bool outerAlias(mlir::DenseSet<mlir::BlockArgument> allOuter) const;
};

// Compute relative access, fail if non-strided (or operation not supported)
mlir::Optional<RelativeAccessPattern>
computeRelativeAccess(mlir::Operation *op, BlockArgumentBoundaryFn fn);

mlir::Optional<RelativeAccessPattern> computeRelativeAccess(mlir::Operation *op,
                                                            mlir::Block *block);

bool hasPerfectAliasing(
    const RelativeAccessPattern &aRap, RelativeAccessPattern bRap,
    const mlir::DenseMap<mlir::BlockArgument, mlir::BlockArgument> &bToA);

// Compute the number of cache misses for a given tile dimensions and strides
// cacheElems = cache size in elements = cache size / element width in bytes
// tileDimensions = dimensions of the tile to be loaded into cache
// tensorStrides = natural / flat strides of the untiled tensor
double computeCacheMiss(double cacheElems,
                        mlir::SmallVector<int64_t, 4> tileDimensions,
                        mlir::SmallVector<int64_t, 4> tensorStrides);

} // namespace pmlc::dialect::pxa
