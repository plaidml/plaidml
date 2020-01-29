// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/IR/Dialect.h"

namespace pmlc::dialect::stdx {

class Dialect : public mlir::Dialect {
 public:
  explicit Dialect(mlir::MLIRContext* ctx);
  static llvm::StringRef getDialectNamespace();
};

}  // namespace pmlc::dialect::stdx
