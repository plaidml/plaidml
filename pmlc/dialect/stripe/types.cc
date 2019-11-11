// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/types.h"

namespace pmlc::dialect::stripe {

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

  static TensorTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key) {
    return new (allocator.allocate<TensorTypeStorage>())
        TensorTypeStorage(std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key));
  }

  Type elementType;
  std::vector<TensorDim> shape;
  OffsetsMap offsets;
  bool is_const;
};

TensorType TensorType::get(           //
    Type elementType,                 //
    llvm::ArrayRef<TensorDim> shape,  //
    const OffsetsMap& offsets,        //
    bool is_const) {
  return Base::get(elementType.getContext(), Types::Tensor, elementType, shape, offsets, is_const);
}

Type TensorType::getElementType() const { return getImpl()->elementType; }

int64_t TensorType::getRank() const { return getImpl()->shape.size(); }

ArrayRef<TensorDim> TensorType::getShape() const { return getImpl()->shape; }

const OffsetsMap& TensorType::getOffsets() const { return getImpl()->offsets; }

bool TensorType::is_const() const { return getImpl()->is_const; }

uint64_t TensorType::getByteSize() const {
  auto scalarType = getElementType().dyn_cast<eltwise::ScalarType>();
  uint64_t total = 0;
  for (const auto& dim : getShape()) {
    if (!dim.size) {
      return 0;
    }
    if (dim.stride > 0) {
      total += (dim.size - 1) * dim.stride;
    }
  }
  return (total + 1) * byte_width(scalarType.type());
}

struct TensorRefTypeStorage : public mlir::TypeStorage {
  TensorRefTypeStorage(Type elementType, int64_t rank, bool is_const, ArrayRef<TensorDim> inShape)
      : elementType(elementType), rank(rank), is_const(is_const) {
    std::copy(inShape.begin(), inShape.end(), std::back_inserter(shape));
  }

  using KeyTy = std::tuple<Type, int64_t, bool, ArrayRef<TensorDim>>;
  bool operator==(const KeyTy& key) const { return key == KeyTy(elementType, rank, is_const, shape); }
  static llvm::hash_code hashKey(const KeyTy& key) { return hash_value(key); }

  static TensorRefTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key) {
    return new (allocator.allocate<TensorRefTypeStorage>())
        TensorRefTypeStorage(std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key));
  }

  Type elementType;
  int64_t rank;
  bool is_const;
  llvm::SmallVector<TensorDim, 4> shape;
};

TensorRefType TensorRefType::get(Type elementType, int64_t rank, bool is_const, ArrayRef<TensorDim> shape) {
  return Base::get(elementType.getContext(), Types::TensorRef, elementType, rank, is_const, shape);
}

TensorRefType TensorRefType::get(TensorType type, bool propagateShape) {
  return Base::get(type.getContext(), Types::TensorRef, type.getElementType(), type.getRank(), type.is_const(),
                   propagateShape ? type.getShape() : ArrayRef<TensorDim>());
}

Type TensorRefType::getElementType() const { return getImpl()->elementType; }

int64_t TensorRefType::getRank() const { return getImpl()->rank; }

bool TensorRefType::is_const() const { return getImpl()->is_const; }

const ArrayRef<TensorDim> TensorRefType::getShape() const { return getImpl()->shape; }

}  // namespace pmlc::dialect::stripe
