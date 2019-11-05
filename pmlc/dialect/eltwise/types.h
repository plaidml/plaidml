// Copyright 2019, Intel Corporation

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "mlir/IR/Types.h"

#include "tile/base/shape.h"

namespace pmlc {
namespace dialect {
namespace eltwise {

using DataType = vertexai::tile::DataType;

enum Kinds {
  // A scalar type
  Scalar = mlir::Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
};

struct ScalarTypeStorage : public mlir::TypeStorage {
  explicit ScalarTypeStorage(DataType type) : type(type) {}

  using KeyTy = int;
  bool operator==(const KeyTy& key) const { return static_cast<int>(type) == key; }
  static llvm::hash_code hashKey(const KeyTy& key) { return key; }
  static KeyTy getKey(DataType type) { return static_cast<int>(type); }

  static ScalarTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key) {
    return new (allocator.allocate<ScalarTypeStorage>()) ScalarTypeStorage(static_cast<DataType>(key));
  }

  DataType type;
};

class ScalarType : public mlir::Type::TypeBase<ScalarType, mlir::Type, ScalarTypeStorage> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == Kinds::Scalar; }

  static ScalarType get(mlir::MLIRContext* context, DataType type) {  //
    return Base::get(context, Kinds::Scalar, type);
  }

  DataType type() { return getImpl()->type; }
};

}  // namespace eltwise
}  // namespace dialect
}  // namespace pmlc
