// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/Types.h"

namespace pmlc::dialect::tile {

class AffineTensorMapType
    : public mlir::Type::TypeBase<AffineTensorMapType, mlir::Type,
                                  mlir::TypeStorage> {
public:
  using Base::Base;
  static AffineTensorMapType get(mlir::MLIRContext *context);
};

class AffineMapType : public mlir::Type::TypeBase<AffineMapType, mlir::Type,
                                                  mlir::TypeStorage> {
public:
  using Base::Base;
  static AffineMapType get(mlir::MLIRContext *context);
};

class AffineConstraintsType
    : public mlir::Type::TypeBase<AffineConstraintsType, mlir::Type,
                                  mlir::TypeStorage> {
public:
  using Base::Base;
  static AffineConstraintsType get(mlir::MLIRContext *context);
};

class StringType
    : public mlir::Type::TypeBase<StringType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static StringType get(mlir::MLIRContext *context);
};

} // namespace pmlc::dialect::tile
