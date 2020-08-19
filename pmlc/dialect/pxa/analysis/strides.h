// Copyright 2020, Intel Corporation

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"

#include "pmlc/dialect/pxa/analysis/affine_expr.h"
#include "pmlc/dialect/pxa/ir/ops.h"

namespace mlir {

// Get the step for a block argument as an IV of an affine.for or
// affine.parallel
int64_t getIVStep(BlockArgument arg);

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

  explicit StrideRange(BlockArgument arg);

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

  int64_t count() {
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

// StrideInfo provides a simple 'stride' multiplier for each affine induction
// variable (from an affine.for or affine.parallel).  Basically, each step of
// the loop moves a pure affine expression by a fixed distance, 'strides' holds
// that distance.  Additionally it holds a fixed offset.
struct StrideInfo {
  int64_t offset;
  DenseMap<BlockArgument, int64_t> strides;

  explicit StrideInfo(BlockArgument arg) : offset(0), strides({{arg, 1}}) {}
  explicit StrideInfo(int64_t offset = 0) : offset(offset) {}

  bool operator==(const StrideInfo &rhs) const {
    return offset == rhs.offset && strides == rhs.strides;
  }
  StrideInfo &operator*=(int64_t factor);
  StrideInfo &operator+=(const StrideInfo &rhs);
  StrideInfo operator+(const StrideInfo &rhs) const {
    StrideInfo r = *this;
    r += rhs;
    return r;
  }

  // Compute the outer and inner portion of stride info with respect to a given
  // block.
  StrideInfo outer(Block *block);
  StrideInfo inner(Block *block);

  // Return the range of a given stride info if it's computable
  StrideRange range() const;

  // Convert a StrideInfo back into an affine expression
  AffineValueExpr toValueExpr(MLIRContext *ctx) const;

  void print(raw_ostream &os, Block *relative = nullptr) const;
};

// Convert a vector of StrideInfo's into a value map
AffineValueMap convertToValueMap(MLIRContext *ctx, ArrayRef<StrideInfo> dims);

// Compute stride info for a given affine value (such an an induction variable
// or the result of an affine.apply). Return None if the expression is not a
// pure affine expression or if any of the gathered strides would be symbolic
Optional<StrideInfo> computeStrideInfo(Value expr);

// Compute stride info but for an affine expression over some set of values
Optional<StrideInfo> computeStrideInfo(AffineExpr expr, ValueRange args);

// Compute 'dimensionalized' strides for a given affine map and arguments
Optional<llvm::SmallVector<StrideInfo, 4>> computeStrideInfo(AffineMap map,
                                                             ValueRange args);

// Compute stride info as additionaly applied to a memRef.
Optional<StrideInfo> computeStrideInfo(MemRefType memRef, AffineMap map,
                                       ValueRange values);

// Helper that works on a affine load / store, etc.
Optional<StrideInfo> computeStrideInfo(pmlc::dialect::pxa::PxaLoadOp op);
Optional<StrideInfo> computeStrideInfo(pmlc::dialect::pxa::PxaReduceOp op);
Optional<StrideInfo> computeStrideInfo(pmlc::dialect::pxa::PxaVectorLoadOp op);
Optional<StrideInfo>
computeStrideInfo(pmlc::dialect::pxa::PxaVectorReduceOp op);

// For a given block + memory access:
// 1) What is the effect of all block args outside and including the
// block on the memory access, in terms of strides.
// 2) For all interior blocks/load width/etc, what is the range of elements
// accessed.

struct RelativeAccessPattern {
  // For each dimension on the access, what are it's strides relative to various
  // block args
  SmallVector<StrideInfo, 4> outer;
  // For each dimension the offset inside the block
  SmallVector<StrideInfo, 4> inner;
  // For each dimension what is the number of accesses
  SmallVector<int64_t, 4> innerCount;
  // For each dimension what is the minimal stride of the access.  Note:
  // dimensions with a count of 1 have a stride of 1 automatically
  SmallVector<int64_t, 4> innerStride;
};

// Compute relative access, fail if non-strided (or operation not supported)
Optional<RelativeAccessPattern> computeRelativeAccess(Block *block,
                                                      Operation *op);

// A StrideArray contains a set of constant factors and a constant offset.
struct StrideArray {
  int64_t offset;
  SmallVector<int64_t, 8> strides;

  explicit StrideArray(unsigned numDims, int64_t offset = 0);
  StrideArray &operator*=(int64_t factor);
  StrideArray &operator+=(const StrideArray &rhs);

  void print(raw_ostream &os);
};

// Compute the StrideArray for a given AffineMap. The map must have a single
// AffineExpr result and this result must be purely affine.
Optional<StrideArray> computeStrideArray(AffineMap map);

} // namespace mlir
