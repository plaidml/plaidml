// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/Dialect.h"

namespace pmlc::dialect::pxa {

class Dialect : public mlir::Dialect {
public:
  explicit Dialect(mlir::MLIRContext *ctx);
  static llvm::StringRef getDialectNamespace();
};

} // namespace pmlc::dialect::pxa
