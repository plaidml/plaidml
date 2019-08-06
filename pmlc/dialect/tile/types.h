// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/Types.h"

namespace pmlc {
namespace dialect {
namespace tile {

enum Kinds {
  AffineIndexMap = mlir::Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_2_TYPE,
  AffineSizeMap,
  String,
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

class StringType : public mlir::Type::TypeBase<StringType, mlir::Type> {
 public:
  using Base::Base;
  static StringType get(mlir::MLIRContext* context);
  static bool kindof(unsigned kind) { return kind == Kinds::String; }
};

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
