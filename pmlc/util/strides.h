
// Copyright 2020, Intel Corporation

#pragma once

#include <ostream>

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/raw_ostream.h"

namespace pmlc::util {

// A StrideArray contains a set of constant factors and a constant offset.
struct StrideArray {
  int64_t offset;
  mlir::SmallVector<int64_t, 8> strides;

  explicit StrideArray(unsigned numDims, int64_t offset = 0);
  StrideArray &operator*=(int64_t factor);
  StrideArray &operator+=(const StrideArray &rhs);

  // void print(mlir::raw_ostream &os);
};

std::ostream &operator<<(std::ostream &os, const StrideArray &val);

// Compute the StrideArray for a given AffineMap. The map must have a single
// AffineExpr result and this result must be purely affine.
mlir::Optional<StrideArray> computeStrideArray(mlir::AffineMap map);

// Compute the StrideArray for a given AffineMap accessing a given memref.
mlir::Optional<StrideArray> computeStrideArray(mlir::MemRefType memRefType,
                                               mlir::AffineMap map);

// Compute the StrideArray for a given AffineMap accessing a given memref.
mlir::Optional<StrideArray> computeStrideArray(mlir::TensorType tensorType,
                                               mlir::AffineMap map);

} // namespace pmlc::util
