// Copyright 2019, Intel Corporation

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "pmlc/dialect/eltwise/types.h"
#include "pmlc/dialect/stripe/mlir.h"
#include "tile/base/shape.h"

namespace pmlc {
namespace dialect {
namespace stripe {

namespace Types {
enum Kinds {
  // An affine is a affine polynomial of indexes over integers
  Affine = Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_1_TYPE,
  // A hardware device identifier
  DeviceID,
  // A hardware device path
  DevicePath,
  // A PRNG state
  Prng,
  // A fully-sized tensor with a memory layout
  Tensor,
  // A tensor reference
  TensorRef,
};
}  // namespace Types

// The Affine type represents an affine expression of indexes
// The type itself is trival, and actual expression is constructed on demand
class AffineType : public Type::TypeBase<AffineType, Type> {
 public:
  using Base::Base;
  static bool kindof(unsigned kind) { return kind == Types::Affine; }
  static AffineType get(MLIRContext* context) { return Base::get(context, Types::Affine); }
};

struct TensorDim {
  // The size of a dimension, or 0 if the dimensions size is not fixed
  int64_t size;
  // The stride of a dimension in linear memory, or 0 if not yet known
  int64_t stride;

  bool operator==(const TensorDim& rhs) const {  //
    return size == rhs.size && stride == rhs.stride;
  }
};

inline llvm::hash_code hash_value(const TensorDim& td) {  //
  return llvm::hash_combine(td.size, td.stride);
}

struct TensorTypeStorage : public mlir::TypeStorage {
  TensorTypeStorage(Type elementType, llvm::ArrayRef<TensorDim> shape, bool is_const)
      : elementType(elementType), shape(shape), is_const(is_const) {}

  using KeyTy = std::tuple<Type, std::vector<TensorDim>, bool>;
  bool operator==(const KeyTy& key) const {
    return elementType == std::get<0>(key) && shape == std::get<1>(key) && is_const == std::get<2>(key);
  }
  static llvm::hash_code hashKey(const KeyTy& key) { return hash_value(key); }

  static TensorTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key) {  // NOLINT
    return new (allocator.allocate<TensorTypeStorage>())
        TensorTypeStorage(std::get<0>(key), std::get<1>(key), std::get<2>(key));
  }

  Type elementType;
  std::vector<TensorDim> shape;
  bool is_const;
};

struct TensorTypeBase {
  /// Return the element type.
  virtual Type getElementType() const = 0;

  /// Return the rank.
  virtual int64_t getRank() const = 0;
};

class TensorType : public Type::TypeBase<TensorType, Type, TensorTypeStorage> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == Types::Tensor; }

  static TensorType get(Type elementType, llvm::ArrayRef<TensorDim> shape, bool is_const) {
    return Base::get(elementType.getContext(), Types::Tensor, elementType, shape, is_const);
  }

  /// Return the element type.
  Type getElementType() const { return getImpl()->elementType; }

  /// Return the rank.
  int64_t getRank() const { return getImpl()->shape.size(); }

  /// Return the shape.
  llvm::ArrayRef<TensorDim> getShape() const { return getImpl()->shape; }

  /// Check if things are const
  bool is_const() const { return getImpl()->is_const; }
};

struct TensorRefTypeStorage : public mlir::TypeStorage {
  TensorRefTypeStorage(Type elementType, size_t rank, bool is_const)
      : elementType(elementType), rank(rank), is_const(is_const) {}

  using KeyTy = std::tuple<Type, size_t, bool>;
  bool operator==(const KeyTy& key) const {
    return elementType == std::get<0>(key) && rank == std::get<1>(key) && is_const == std::get<2>(key);
  }
  static llvm::hash_code hashKey(const KeyTy& key) { return hash_value(key); }

  static TensorRefTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key) {  // NOLINT
    return new (allocator.allocate<TensorRefTypeStorage>())
        TensorRefTypeStorage(std::get<0>(key), std::get<1>(key), std::get<2>(key));
  }

  Type elementType;
  size_t rank;
  bool is_const;
};

class TensorRefType : public Type::TypeBase<TensorRefType, Type, TensorRefTypeStorage> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == Types::TensorRef; }

  static TensorRefType get(Type elementType, size_t rank, bool is_const) {
    return Base::get(elementType.getContext(), Types::TensorRef, elementType, rank, is_const);
  }

  /// Return the element type.
  Type getElementType() const { return getImpl()->elementType; }

  /// Return the rank.
  int64_t getRank() const { return getImpl()->rank; }

  /// Check if things are const
  bool is_const() const { return getImpl()->is_const; }
};

// A PRNG state.
class PrngType : public Type::TypeBase<PrngType, Type> {
 public:
  using Base::Base;
  static bool kindof(unsigned kind) { return kind == Types::Prng; }
  static PrngType get(MLIRContext* context) { return Base::get(context, Types::Prng); }
};

// A relative identifier for a hardware component capable of storing tensor data or executing a block of
// instructions.
class DeviceIDType : public Type::TypeBase<DeviceIDType, Type> {
 public:
  using Base::Base;
  static bool kindof(unsigned kind) { return kind == Types::DeviceID; }
  static DeviceIDType get(MLIRContext* context) { return Base::get(context, Types::DeviceID); }
};

// An absolute path to a hardware component capable of storing tensor data or executing a block of
// instructions.
class DevicePathType : public Type::TypeBase<DevicePathType, Type> {
 public:
  using Base::Base;
  static bool kindof(unsigned kind) { return kind == Types::DevicePath; }
  static DevicePathType get(MLIRContext* context) { return Base::get(context, Types::DevicePath); }
};

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
