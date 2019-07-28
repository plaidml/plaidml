// Copyright 2019, Intel Corporation

#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/StandardOps/Ops.h"

#include "pmlc/dialect/scalar/ops.h"

#define DEBUG_TYPE "pml_scalar"

namespace pmlc {
namespace dialect {
namespace scalar {

using vertexai::tile::DataTypeFromString;

class Dialect : public mlir::Dialect {
 public:
  explicit Dialect(mlir::MLIRContext* ctx) : mlir::Dialect("pml_scalar", ctx) {
    addTypes<ScalarType>();
    addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/scalar/ops.cpp.inc"
        >();
  }

  mlir::Type parseType(llvm::StringRef spec, mlir::Location loc) const override {  //
    return ScalarType::get(getContext(), DataTypeFromString(spec));
  }

  void printType(mlir::Type type, llvm::raw_ostream& os) const override {
    if (auto t = type.dyn_cast<ScalarType>()) {
      os << to_string(t.type());
    } else {
      llvm_unreachable("unhandled scalar type");
    }
  }

  mlir::Operation* materializeConstant(  //
      mlir::OpBuilder& builder,          //
      mlir::Attribute value,             //
      mlir::Type type,                   //
      mlir::Location loc) override {
    return builder.create<ScalarConstantOp>(loc, type, value);
  }
};

// static mlir::DialectRegistration<mlir::StandardOpsDialect> StandardOps;
static mlir::DialectRegistration<Dialect> ScalarOps;

}  // namespace scalar
}  // namespace dialect
}  // namespace pmlc
