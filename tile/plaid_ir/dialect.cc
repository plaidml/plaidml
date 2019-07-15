// Copyright 2019, Intel Corporation

#include "tile/plaid_ir/dialect.h"
#include "tile/plaid_ir/ops.h"
#include "tile/plaid_ir/types.h"

namespace vertexai {
namespace tile {
namespace plaid_ir {

PlaidDialect::PlaidDialect(mlir::MLIRContext* ctx) : Dialect("plaid", ctx) {
  addTypes<TensorType>();
  addTypes<AffineType>();
  addOperations<
#define GET_OP_LIST
#include "tile/plaid_ir/ops.cpp.inc"
      >();
  printf("Got to PlaidDialect::PlaidDialect\n");
}

mlir::Type PlaidDialect::parseType(llvm::StringRef tyData, mlir::Location loc) const {
  throw std::runtime_error("Unimplemented");
}

static void print(TensorType t, llvm::raw_ostream& os) {
  os << "tensor ";
  os << t.base() << "(";
  bool first = true;
  for (const auto& dim : t.dims()) {
    if (!first) {
      os << ", ";
    }
    first = false;
    if (dim.unit != "") {
      os << dim.unit << ":";
    }
    if (dim.size == 0) {
      os << "?";
    } else {
      os << std::to_string(dim.size);
    }
    if (dim.stride != 0) {
      os << ":" << std::to_string(dim.stride);
    }
    if (dim.name != "") {
      os << ":" << dim.name;
    }
  }
  os << ")";
}

static void print(AffineType, llvm::raw_ostream& os) { os << "affine"; }

void PlaidDialect::printType(mlir::Type type, llvm::raw_ostream& os) const {
  if (auto t = type.dyn_cast<TensorType>()) {
    print(t, os);
  } else if (auto t = type.dyn_cast<AffineType>()) {
    print(t, os);
  } else {
    llvm_unreachable("unhandled Plaid type");
  }
}

}  // namespace plaid_ir
}  // namespace tile
}  // namespace vertexai
