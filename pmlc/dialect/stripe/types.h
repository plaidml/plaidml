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

namespace PlaidKind {
enum Kinds {
  // An affine is a affine polynomial of indexes over integers
  Affine = Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_1_TYPE,
  // A tensor reference
  Tensor,
  // A PRNG state
  Prng,
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

struct TensorTypeStorage : public mlir::TypeStorage {
  TensorTypeStorage(eltwise::ScalarType base, size_t ndim) : base(base), ndim(ndim) {}

  using KeyTy = std::tuple<eltwise::ScalarType, size_t>;
  bool operator==(const KeyTy& key) const { return base == std::get<0>(key) && ndim == std::get<1>(key); }
  static llvm::hash_code hashKey(const KeyTy& key) { return hash_value(key); }

  static TensorTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key) {  // NOLINT
    return new (allocator.allocate<TensorTypeStorage>()) TensorTypeStorage(std::get<0>(key), std::get<1>(key));
  }

  eltwise::ScalarType base;
  size_t ndim;
};

class TensorType : public Type::TypeBase<TensorType, Type, TensorTypeStorage> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == PlaidKind::Tensor; }

  static TensorType get(MLIRContext* context, eltwise::ScalarType base, size_t ndim) {
    return Base::get(context, PlaidKind::Tensor, base, ndim);
  }

  eltwise::ScalarType base() { return getImpl()->base; }
  size_t ndim() { return getImpl()->ndim; }
};

// A PRNG state.
class PrngType : public Type::TypeBase<PrngType, Type> {
 public:
  using Base::Base;
  static bool kindof(unsigned kind) { return kind == PlaidKind::Prng; }
  static PrngType get(MLIRContext* context) { return Base::get(context, PlaidKind::Prng); }
};

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
