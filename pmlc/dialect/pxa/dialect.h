// Copyright 2019, Intel Corporation

#pragma once

#include <string>

#include "mlir/IR/Dialect.h"

namespace pmlc::dialect::pxa {

class Dialect : public mlir::Dialect {
 public:
  explicit Dialect(mlir::MLIRContext* ctx) : mlir::Dialect("pxa", ctx) {}
  static llvm::StringRef getDialectNamespace() { return "pxa"; }
};

}  // namespace pmlc::dialect::pxa
