// Copyright 2020, Intel Corporation

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"

#include "pmlc/dialect/pxa/ir/ops.h"

namespace mlir {

struct StrideRange {
  bool valid;
  int64_t minVal;
  int64_t maxVal;
  int64_t stride;

  explicit StrideRange(int64_t val)
      : valid(true), minVal(val), maxVal(val), stride(0) {}
  explicit StrideRange(BlockArgument arg);
  StrideRange &operator*=(int64_t factor);
  StrideRange operator*(int64_t factor) const {
    StrideRange r = *this;
    r *= factor;
    return r;
  }
  StrideRange &operator+=(const StrideRange &rhs);
  StrideRange operator+(const StrideRange &rhs) const {
    StrideRange r = *this;
    r += rhs;
    return r;
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

  explicit StrideInfo(int64_t offset = 0) : offset(offset) {}

  bool operator==(const StrideInfo &rhs) const {
    return offset == rhs.offset && strides == rhs.strides;
  }
  StrideInfo &operator*=(int64_t factor);
  StrideInfo &operator+=(const StrideInfo &rhs);

  // Compute the outer and inner portion of stride info with respect to a given
  // block.
  StrideInfo outer(Block *block);
  StrideInfo inner(Block *block);

  // Return the range of a given stride info if it's computable
  StrideRange range() const;

  AffineExpr toExpr(MLIRContext *ctx, ValueRange operands) const;

  void print(raw_ostream &os, Block *relative = nullptr) const;
};

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
Optional<StrideInfo> computeStrideInfo(AffineLoadOp op);
Optional<StrideInfo> computeStrideInfo(AffineStoreOp op);
Optional<StrideInfo> computeStrideInfo(pmlc::dialect::pxa::AffineReduceOp op);

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
