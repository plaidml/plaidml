// Copyright 2019, Intel Corporation

#include "pmlc/dialect/eltwise/dialect.h"

#include <utility>

#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"

#include "pmlc/dialect/eltwise/ops.h"

#define DEBUG_TYPE "eltwise"

namespace pmlc {
namespace dialect {
namespace eltwise {

using vertexai::tile::DataTypeFromString;

namespace {

struct OpAsmInterface : public mlir::OpAsmDialectInterface {
  using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

  /// Get a special name to use when printing the given operation. The desired
  /// name should be streamed into 'os'.
  void getOpResultName(Operation* op, llvm::raw_ostream& os) const final {  // NOLINT
    if (auto str_attr = op->getAttrOfType<mlir::StringAttr>("scalar_name")) {
      std::string s = str_attr.getValue().str();
      os << "s_" << s.substr(1);
    } else if (auto const_op = llvm::dyn_cast<ScalarConstantOp>(op)) {
      auto attr = const_op.value();
      if (auto int_attr = attr.dyn_cast<IntegerAttr>()) {
        os << 'c' << int_attr.getValue();
      } else {
        os << "cst";
      }
    }
  }
  void getTypeAliases(mlir::SmallVectorImpl<std::pair<Type, StringRef>>& aliases) const final {  // NOLINT
    for (const auto dt : vertexai::tile::GetDataTypeSet()) {
      // Intern the string
      auto id = mlir::Identifier::get(to_string(dt), getDialect()->getContext());
      // Get the type
      Type t = RankedTensorType::get({}, ScalarType::get(getDialect()->getContext(), dt));
      // Add the alias
      aliases.push_back(std::make_pair(t, id));
    }
  }
};

}  // namespace

Dialect::Dialect(mlir::MLIRContext* ctx) : mlir::Dialect(getDialectNamespace(), ctx) {
  addTypes<ScalarType>();
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/eltwise/ops.cc.inc"
      >();
  addInterfaces<OpAsmInterface>();
}

std::string Dialect::getCanonicalOpName(llvm::StringRef name) {
  if (name == "cond") {
    name = "select";
  }
  return llvm::formatv("{0}.{1}", getDialectNamespace(), name).str();
}

mlir::Type Dialect::parseType(llvm::StringRef spec, mlir::Location loc) const {  //
  return ScalarType::get(getContext(), DataTypeFromString(spec));
}

void Dialect::printType(mlir::Type type, llvm::raw_ostream& os) const {
  if (auto t = type.dyn_cast<ScalarType>()) {
    os << to_string(t.type());
  } else {
    llvm_unreachable("unhandled scalar type");
  }
}

mlir::Operation* Dialect::materializeConstant(  //
    mlir::OpBuilder& builder,                   //
    mlir::Attribute value,                      //
    mlir::Type type,                            //
    mlir::Location loc) {
  return builder.create<ScalarConstantOp>(loc, type, value);
}

static mlir::DialectRegistration<Dialect> EltwiseOps;

}  // namespace eltwise
}  // namespace dialect
}  // namespace pmlc
