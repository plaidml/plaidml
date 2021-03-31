// Copyright 2020 Intel Corporation

#include "pmlc/util/strides.h"

#include <vector>

#include "mlir/Analysis/AffineStructures.h"
#include "pmlc/util/logging.h"

namespace pmlc::util {

StrideArray::StrideArray(unsigned numDims, int64_t offset)
    : offset(offset), strides(numDims) {}

StrideArray &StrideArray::operator*=(int64_t factor) {
  offset *= factor;
  for (auto &dim : strides)
    dim *= factor;
  return *this;
}

StrideArray &StrideArray::operator+=(const StrideArray &rhs) {
  assert(strides.size() == rhs.strides.size() && "strides sizes much match");
  offset += rhs.offset;
  for (unsigned i = 0, e = strides.size(); i < e; ++i) {
    strides[i] += rhs.strides[i];
  }
  return *this;
}

std::ostream &operator<<(std::ostream &os, const StrideArray &val) {
  os << val.offset << ":[";
  for (auto item : llvm::enumerate(val.strides)) {
    if (item.index())
      os << ", ";
    os << item.value();
  }
  os << ']';
  return os;
}

llvm::Optional<StrideArray> computeStrideArray(mlir::AffineMap map) {
  assert(map.getNumResults() == 1);
  std::vector<llvm::SmallVector<int64_t, 8>> flat;
  if (failed(getFlattenedAffineExprs(map, &flat, nullptr)))
    return llvm::None;

  StrideArray ret(map.getNumDims(), flat.front().back());
  for (unsigned i = 0, e = map.getNumDims(); i < e; i++) {
    ret.strides[i] = flat.front()[i];
  }

  return ret;
}

mlir::Optional<StrideArray> computeStrideArray(mlir::MemRefType memRefType,
                                               mlir::AffineMap map) {
  assert(map.getNumResults() == memRefType.getRank());

  // MLIR doesnt' corrently handle rank 0 in some places, early exit
  if (memRefType.getRank() == 0) {
    return StrideArray(map.getNumDims(), 0);
  }

  // Get the memref strides
  int64_t offset;
  llvm::SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(memRefType, strides, offset)))
    return llvm::None;

  // Get the dimensionalized map multipliers
  std::vector<llvm::SmallVector<int64_t, 8>> flat;
  if (failed(getFlattenedAffineExprs(map, &flat, nullptr))) {
    return llvm::None;
  }

  // Flatten the per-dimension data via memory order
  StrideArray ret(map.getNumDims(), offset);
  for (unsigned d = 0; d < memRefType.getRank(); d++) {
    StrideArray perDim(map.getNumDims(), flat[d].back());
    for (unsigned i = 0, e = map.getNumDims(); i < e; i++) {
      perDim.strides[i] = flat[d][i];
    }
    perDim *= strides[d];
    ret += perDim;
  }
  return ret;
}

mlir::Optional<StrideArray> computeStrideArray(mlir::TensorType tensorType,
                                               mlir::AffineMap map) {
  auto memRefType =
      mlir::MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  return computeStrideArray(memRefType, map);
}

} // namespace pmlc::util
