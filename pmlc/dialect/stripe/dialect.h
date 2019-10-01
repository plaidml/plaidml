// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/Dialect.h"

namespace pmlc {
namespace dialect {
namespace stripe {

class Dialect : public mlir::Dialect {
 public:
  explicit Dialect(mlir::MLIRContext* ctx);

  static llvm::StringRef getDialectNamespace() { return "stripe"; }

  mlir::Type parseTensor(llvm::StringRef tyData, mlir::Location loc) const;
  mlir::Type parseTensorRef(llvm::StringRef tyData, mlir::Location loc) const;

  mlir::Type parseType(llvm::StringRef tyData, mlir::Location loc) const override;
  void printType(mlir::Type type, llvm::raw_ostream& os) const override;
};

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
