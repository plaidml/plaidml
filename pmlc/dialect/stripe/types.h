// Copyright 2019, Intel Corporation

#pragma once

#include <algorithm>
#include <tuple>
#include <utility>
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
  // An executor (entity that can evaluate computations)
  Executor,
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

// Executor represents an entity capable of running code within a block.
// This type is used as the element type for tensors describing a set of execution units.
class ExecutorType : public Type::TypeBase<ExecutorType, Type> {
 public:
  using Base::Base;
  static bool kindof(unsigned kind) { return kind == Types::Executor; }
  static ExecutorType get(MLIRContext* context) { return Base::get(context, Types::Executor); }
};

// Describes one dimension of a tensor.
struct TensorDim {
  // The size of a dimension, or 0 if the dimension's size is not fixed.
  int64_t size;

  // The stride of a dimension (may be zero).
  int64_t stride;

  // The hardware class identified by this dimension.
  llvm::StringRef cls;

  bool operator==(const TensorDim& rhs) const {  //
    return std::tie(size, stride, cls) == std::tie(rhs.size, rhs.stride, rhs.cls);
  }
};

inline llvm::hash_code hash_value(const TensorDim& td) {  //
  return llvm::hash_combine(td.size, td.stride);
}

using OffsetsMap = llvm::SmallDenseMap<mlir::Identifier, int64_t>;

struct TensorTypeStorage : public mlir::TypeStorage {
  TensorTypeStorage(Type elementType, llvm::ArrayRef<TensorDim> shape, OffsetsMap offsets, bool is_const)
      : elementType(elementType), shape(shape), offsets(offsets), is_const(is_const) {}

  using KeyTy = std::tuple<Type, std::vector<TensorDim>, OffsetsMap, bool>;

  bool operator==(const KeyTy& key) const { return std::tie(elementType, shape, offsets, is_const) == key; }

  static llvm::hash_code hashKey(const KeyTy& key) {
    std::vector<std::pair<const void*, int64_t>> offset_vec;
    const OffsetsMap& offsets = std::get<2>(key);
    offset_vec.reserve(offsets.size());
    for (const auto& id_offset : offsets) {
      offset_vec.emplace_back(id_offset.first.getAsOpaquePointer(), id_offset.second);
    }
    std::sort(offset_vec.begin(), offset_vec.end());
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key), llvm::hash_combine(offset_vec), std::get<3>(key));
  }

  static TensorTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key) {  // NOLINT
    return new (allocator.allocate<TensorTypeStorage>())
        TensorTypeStorage(std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key));
  }

  Type elementType;
  std::vector<TensorDim> shape;
  OffsetsMap offsets;
  bool is_const;
};

// The PlaidML Tensor type.
//
// Tensors consist of an element type, a shape, and an offset table; the shape is a sequence of dimensions.
//
// Early in compilation, we might not have any information at all about a tensor dimension beyond its
// existance (i.e. we might know the rank of the tensor, but little else).  Over the course of compilation,
// the tensor's dimensions will be fixed to particular sizes; as the tensor is laid out for the target system,
// the tensor's shape and offsets will map tuples of tensor indicies to particular hardware units, e.g. "(3,
// 7, 0, 10, 0) => chip #3, compute unit #7, memory bank #0, bytes [1000-1004)".  (Note that these may be
// either logical or physical indicies, depending on the target system).
//
// So when fully specified, each tensor dimension will have a size, a hardware class identifier, and a stride.
// To resolve an index tuple (i.e. mapping the index tuple to a particular element), the index value is
// multiplied by the stride, and accumulated into the index for the hardware class, which starts at the offset
// specified for that hardware class in the offsets table (or zero if the hardware class isn't present in the
// offsets table).
//
// By convention, the hardware class identifier for linear address space is "address"; this is also the
// identifier that should be used prior to hardware assignment.
class TensorType : public Type::TypeBase<TensorType, Type, TensorTypeStorage> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == Types::Tensor; }

  static TensorType get(                //
      Type elementType,                 //
      llvm::ArrayRef<TensorDim> shape,  //
      const OffsetsMap& offsets,        //
      bool is_const) {
    return Base::get(elementType.getContext(), Types::Tensor, elementType, shape, offsets, is_const);
  }

  /// Return the element type.
  Type getElementType() const { return getImpl()->elementType; }

  /// Return the rank.
  int64_t getRank() const { return getImpl()->shape.size(); }

  /// Return the shape.
  llvm::ArrayRef<TensorDim> getShape() const { return getImpl()->shape; }

  /// Return the offsets table.
  const OffsetsMap& getOffsets() const { return getImpl()->offsets; }

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

  static TensorRefType get(TensorType type) {
    return Base::get(type.getContext(), Types::TensorRef, type.getElementType(), type.getRank(), type.is_const());
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

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
