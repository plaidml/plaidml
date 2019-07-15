// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/Dialect.h"

namespace vertexai {
namespace tile {
namespace plaid_ir {

class PlaidDialect : public mlir::Dialect {
 public:
  explicit PlaidDialect(mlir::MLIRContext* ctx);
  mlir::Type parseType(llvm::StringRef tyData, mlir::Location loc) const override;
  void printType(mlir::Type type, llvm::raw_ostream& os) const override;
};

}  // namespace plaid_ir
}  // namespace tile
}  // namespace vertexai
