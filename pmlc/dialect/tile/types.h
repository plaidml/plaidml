// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/Types.h"

namespace pmlc {
namespace dialect {
namespace tile {

namespace Kinds {
enum Kind {
  AffineIndexMap = mlir::Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_2_TYPE,
  AffineSizeMap,
  AffineTensorMap,
  AffineMap,
  AffineConstraints,
  String,
};
}

class AffineTensorMapType : public mlir::Type::TypeBase<AffineTensorMapType, mlir::Type> {
 public:
  using Base::Base;
  static AffineTensorMapType get(mlir::MLIRContext* context);
  static bool kindof(unsigned kind) { return kind == Kinds::AffineTensorMap; }
};

class AffineMapType : public mlir::Type::TypeBase<AffineMapType, mlir::Type> {
 public:
  using Base::Base;
  static AffineMapType get(mlir::MLIRContext* context);
  static bool kindof(unsigned kind) { return kind == Kinds::AffineMap; }
};

class AffineIndexMapType : public mlir::Type::TypeBase<AffineIndexMapType, mlir::Type> {
 public:
  using Base::Base;
  static AffineIndexMapType get(mlir::MLIRContext* context);
  static bool kindof(unsigned kind) { return kind == Kinds::AffineIndexMap; }
};

class AffineSizeMapType : public mlir::Type::TypeBase<AffineSizeMapType, mlir::Type> {
 public:
  using Base::Base;
  static AffineSizeMapType get(mlir::MLIRContext* context);
  static bool kindof(unsigned kind) { return kind == Kinds::AffineSizeMap; }
};

class AffineConstraintsType : public mlir::Type::TypeBase<AffineConstraintsType, mlir::Type> {
 public:
  using Base::Base;
  static AffineConstraintsType get(mlir::MLIRContext* context);
  static bool kindof(unsigned kind) { return kind == Kinds::AffineConstraints; }
};

class StringType : public mlir::Type::TypeBase<StringType, mlir::Type> {
 public:
  using Base::Base;
  static StringType get(mlir::MLIRContext* context);
  static bool kindof(unsigned kind) { return kind == Kinds::String; }
};

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
