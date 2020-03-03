// Copyright 2020, Intel Corporation

#pragma once

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"

#include "pmlc/dialect/pxa/ir/ops.h"

namespace mlir {

// StrideInfo provides a simple 'stride' multiplier for each affine induction
// variable (from an affine.for or affine.parallel).  Basically, each step of
// the loop moves a pure affine expression by a fixed distance, 'strides' holds
// that distance.  Additionally it holds a fixed offset.
struct StrideInfo {
  int64_t offset;
  DenseMap<BlockArgument, int64_t> strides;

  explicit StrideInfo(int64_t offset = 0) : offset(offset) {}

  StrideInfo &operator*=(int64_t factor);
  StrideInfo &operator+=(const StrideInfo &rhs);

  void print(raw_ostream &os, Block *relative = nullptr);
};

// Compute stride info for a given affine value (such an an induction variable
// or the result of an affine.apply). Return None if the expression is not a
// pure affine expression or if any of the gathered strides would be symbolic
Optional<StrideInfo> computeStrideInfo(Value expr);

// Compute stride info but for an affine expression over some set of values
Optional<StrideInfo> computeStrideInfo(AffineExpr expr, ValueRange args);

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
