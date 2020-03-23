// Copyright 2019, Intel Corporation

#pragma once

#include <string>

#include "mlir/IR/Dialect.h"

namespace pmlc::dialect::eltwise {

class Dialect : public mlir::Dialect {
public:
  explicit Dialect(mlir::MLIRContext *ctx);

  static llvm::StringRef getDialectNamespace() { return "eltwise"; }
  static std::string getCanonicalOpName(llvm::StringRef name);

  mlir::Operation *materializeConstant(mlir::OpBuilder &builder,
                                       mlir::Attribute value, mlir::Type type,
                                       mlir::Location loc) override;
};

} // namespace pmlc::dialect::eltwise
