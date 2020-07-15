// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/Types.h"

namespace pmlc::dialect::tile {

namespace TypeKinds {
enum Kind {
  AffineMap = mlir::Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_2_TYPE,
  AffineTensorMap,
  AffineConstraints,
  String,
};
}

class AffineTensorMapType
    : public mlir::Type::TypeBase<AffineTensorMapType, mlir::Type,
                                  mlir::TypeStorage> {
public:
  using Base::Base;
  static AffineTensorMapType get(mlir::MLIRContext *context);
  static bool kindof(unsigned kind) {
    return kind == TypeKinds::AffineTensorMap;
  }
};

class AffineMapType : public mlir::Type::TypeBase<AffineMapType, mlir::Type,
                                                  mlir::TypeStorage> {
public:
  using Base::Base;
  static AffineMapType get(mlir::MLIRContext *context);
  static bool kindof(unsigned kind) { return kind == TypeKinds::AffineMap; }
};

class AffineConstraintsType
    : public mlir::Type::TypeBase<AffineConstraintsType, mlir::Type,
                                  mlir::TypeStorage> {
public:
  using Base::Base;
  static AffineConstraintsType get(mlir::MLIRContext *context);
  static bool kindof(unsigned kind) {
    return kind == TypeKinds::AffineConstraints;
  }
};

class StringType
    : public mlir::Type::TypeBase<StringType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static StringType get(mlir::MLIRContext *context);
  static bool kindof(unsigned kind) { return kind == TypeKinds::String; }
};

} // namespace pmlc::dialect::tile
