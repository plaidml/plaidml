// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/IR/Dialect.h"

namespace pmlc::dialect::xsmm {

class Dialect : public mlir::Dialect {
public:
  explicit Dialect(mlir::MLIRContext *ctx);
  static llvm::StringRef getDialectNamespace() { return "xsmm"; }
};

} // namespace pmlc::dialect::xsmm
