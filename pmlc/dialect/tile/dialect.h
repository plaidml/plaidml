// Copyright 2019, Intel Corporation

#pragma once

#include <string>

#include "mlir/IR/Dialect.h"

namespace pmlc {
namespace dialect {
namespace tile {

class Dialect : public mlir::Dialect {
 public:
  explicit Dialect(mlir::MLIRContext* ctx);

  static llvm::StringRef getDialectNamespace() { return "tile"; }
  static std::string getCanonicalOpName(llvm::StringRef name);

  void printType(mlir::Type type, llvm::raw_ostream& os) const override;

  mlir::Operation* materializeConstant(  //
      mlir::OpBuilder& builder,          //
      mlir::Attribute value,             //
      mlir::Type type,                   //
      mlir::Location loc) override;
};

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
