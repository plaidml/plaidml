// Copyright 2019, Intel Corporation

#pragma once

#include <string>

#include "mlir/IR/Dialect.h"

namespace pmlc {
namespace dialect {
namespace eltwise {

class Dialect : public mlir::Dialect {
 public:
  explicit Dialect(mlir::MLIRContext* ctx);

  static llvm::StringRef getDialectNamespace() { return "eltwise"; }
  static std::string getCanonicalOpName(llvm::StringRef name);

  mlir::Type parseType(llvm::StringRef spec, mlir::Location loc) const override;
  void printType(mlir::Type type, llvm::raw_ostream& os) const override;

  mlir::Operation* materializeConstant(  //
      mlir::OpBuilder& builder,          //
      mlir::Attribute value,             //
      mlir::Type type,                   //
      mlir::Location loc) override;
};

}  // namespace eltwise
}  // namespace dialect
}  // namespace pmlc
