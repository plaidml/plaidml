// Copyright 2018, Intel Corporation

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "pmlc/dialect/mir/mlir.h"
#include "pmlc/dialect/mir/types.h"

namespace pmlc {
namespace dialect {
namespace mir {

namespace PlaidAttrKind {
enum Kinds {
  // A temporary standin for the tensor layout information until MLIR makes one
  TensorLayout = Attribute::Kind::FIRST_PRIVATE_EXPERIMENTAL_1_ATTR,
};
}  // namespace PlaidAttrKind

// First, we define the layout information we track about each dimension
struct TensorDim {
  // The 'unit' specifies which hardware unit this dimension represents, if it
  // is the empty string, it represents a dimension in linear memory
  std::string unit;
  // The size of a dimension, or 0 if the dimensions size is not fixed
  int64_t size;
  // The stride of a dimension in linear memory, or 0 if not yet known
  int64_t stride;

  bool operator==(const TensorDim& rhs) const { return unit == rhs.unit && size == rhs.size && stride == rhs.stride; }
};

inline llvm::hash_code hash_value(const TensorDim& td) { return llvm::hash_combine(td.unit, td.size, td.stride); }

struct TensorLayoutAttrStorage : public mlir::AttributeStorage {
  TensorLayoutAttrStorage(scalar::ScalarType base, const std::vector<TensorDim>& dims) : base(base), dims(dims) {}

  using KeyTy = std::tuple<scalar::ScalarType, std::vector<TensorDim>>;

  bool operator==(const KeyTy& key) const { return base == std::get<0>(key) && dims == std::get<1>(key); }

  static llvm::hash_code hashKey(const KeyTy& key) { return hash_value(key); }

  static TensorLayoutAttrStorage* construct(mlir::AttributeStorageAllocator& allocator, const KeyTy& key) {  // NOLINT
    return new (allocator.allocate<TensorLayoutAttrStorage>())
        TensorLayoutAttrStorage(std::get<0>(key), std::get<1>(key));
  }

  scalar::ScalarType base;
  std::vector<TensorDim> dims;
};

class TensorLayoutAttr : public Attribute::AttrBase<TensorLayoutAttr, Attribute, TensorLayoutAttrStorage> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == PlaidAttrKind::TensorLayout; }

  static TensorLayoutAttr get(MLIRContext* context, scalar::ScalarType base, std::vector<TensorDim> dims = {}) {
    return Base::get(context, PlaidAttrKind::TensorLayout, base, dims);
  }

  scalar::ScalarType base() { return getImpl()->base; }
  const std::vector<TensorDim>& dims() { return getImpl()->dims; }
};

}  // namespace mir
}  // namespace dialect
}  // namespace pmlc
