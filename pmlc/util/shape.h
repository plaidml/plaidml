// Copyright 2020 Intel Corporation

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"

#include "pmlc/util/enums.h"

namespace pmlc::util {

struct TensorShape {
  DataType elementType;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  explicit TensorShape(DataType elementType = DataType::invalid)
      : elementType(elementType) {}

  TensorShape(DataType elementType, llvm::ArrayRef<int64_t> sizes)
      : elementType(elementType), sizes(sizes) {}

  TensorShape(DataType elementType, llvm::ArrayRef<int64_t> sizes,
              llvm::ArrayRef<int64_t> strides)
      : elementType(elementType), sizes(sizes), strides(strides) {}

  static TensorShape fromType(mlir::Type type);

  std::string str() const;
  size_t getRank() const { return sizes.size(); }
  size_t getByteSize() const;

  bool operator==(const TensorShape &rhs) const;
  bool operator!=(const TensorShape &rhs) const { return !(*this == rhs); }
};

using TensorShapes = llvm::SmallVector<TensorShape, 4>;

} // namespace pmlc::util
