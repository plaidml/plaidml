// Copyright 2018, Intel Corporation

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "tile/plaid_ir/mlir.h"

namespace vertexai {
namespace tile {
namespace plaid_ir {

namespace PlaidKind {
enum Kinds {
  // An affine is a affine polynomial of indexes over integers
  Affine = Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_1_TYPE,
  // A tensor represents a given storage location
  Tensor,
};
}  // namespace PlaidKind

// The Affine type represents an affine expression of indexes
// The type itself is trival, and actual expression is constructed on demand
class AffineType : public Type::TypeBase<AffineType, Type> {
 public:
  using Base::Base;
  static bool kindof(unsigned kind) { return kind == PlaidKind::Affine; }
  static AffineType get(MLIRContext* context) { return Base::get(context, PlaidKind::Affine); }
};

// TODO: Integers + Boolean
// 1) Use MLIR's signless integer types?
// 2) Use ngraph dialects integer types?
// 3) Declare yet another integer type?

// PlaidIR uses MLIR's native floating point type
using FloatType = mlir::FloatType;

// Hack for now an use MLIR indexes as integer types for sizing
using IntegerType = mlir::IndexType;

// Plaid IR uses it's own tensor type since it want to track
// additional information (such and layout + location) beyond the MLIR tensor type,
// and subclassing is problematic due to LLVM style RTTI.

// First, we define the information the tensor type tracks about each dimension
struct TensorDim {
  // The 'unit' specifies which hardware unit this dimension represents, if it
  // is the empty string, it represents a dimension in linear memory
  std::string unit;
  // The 'name' of a dimension represents the logical meaning of a dimension.
  std::string name;
  // The size of a dimension, or 0 if the dimensions size is not fixed
  int64_t size;
  // The stride of a dimension in linear memory, or 0 if not yet known
  int64_t stride;

  bool operator==(const TensorDim& rhs) const {
    return unit == rhs.unit && name == rhs.name && size == rhs.size && stride == rhs.stride;
  }
};

inline llvm::hash_code hash_value(const TensorDim& td) {
  return llvm::hash_combine(td.unit, td.name, td.size, td.stride);
}

struct TensorTypeStorage : public mlir::TypeStorage {
  TensorTypeStorage(Type base, const std::vector<TensorDim>& dims, const NamedAttributeList& attrs)
      : base(base), dims(dims), attrs(attrs) {}

  using KeyTy = std::tuple<Type, std::vector<TensorDim>, NamedAttributeList>;

  bool operator==(const KeyTy& key) const {
    NamedAttributeList key_attr = std::get<2>(key);
    return base == std::get<0>(key) && dims == std::get<1>(key) && attrs.getAttrs() == key_attr.getAttrs();
  }

  static llvm::hash_code hashKey(const KeyTy& key) { return hash_value(key); }

  static TensorTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key) {  // NOLINT
    return new (allocator.allocate<TensorTypeStorage>())
        TensorTypeStorage(std::get<0>(key), std::get<1>(key), std::get<2>(key));
  }

  Type base;
  std::vector<TensorDim> dims;
  NamedAttributeList attrs;
};

class TensorType : public Type::TypeBase<TensorType, Type, TensorTypeStorage> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == PlaidKind::Tensor; }

  static TensorType get(MLIRContext* context, Type base, std::vector<TensorDim> dims = {},
                        NamedAttributeList attrs = NamedAttributeList()) {
    return Base::get(context, PlaidKind::Tensor, base, dims, attrs);
  }

  Type base() { return getImpl()->base; }
  const std::vector<TensorDim>& dims() { return getImpl()->dims; }
  const NamedAttributeList& attrs() { return getImpl()->attrs; }
};

}  // namespace plaid_ir
}  // namespace tile
}  // namespace vertexai
