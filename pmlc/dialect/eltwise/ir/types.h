// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/IR/Types.h"

namespace pmlc::dialect::eltwise {

class APFloatType : public mlir::Type::TypeBase<APFloatType, mlir::Type,
                                                mlir::DefaultTypeStorage> {
public:
  using Base::Base;
};

class APSignedIntegerType
    : public mlir::Type::TypeBase<APSignedIntegerType, mlir::Type,
                                  mlir::DefaultTypeStorage> {
public:
  using Base::Base;
};

class APUnsignedIntegerType
    : public mlir::Type::TypeBase<APUnsignedIntegerType, mlir::Type,
                                  mlir::DefaultTypeStorage> {
public:
  using Base::Base;
};

} // namespace pmlc::dialect::eltwise
