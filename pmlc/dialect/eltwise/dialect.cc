// Copyright 2019, Intel Corporation

#include "pmlc/dialect/eltwise/dialect.h"

#include <utility>

#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/DebugStringHelper.h"

#include "base/util/logging.h"
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
  void getOpResultName(Operation* op, llvm::raw_ostream& os) const final {
    if (auto attr = op->getAttrOfType<mlir::StringAttr>("scalar_name")) {
      os << "s_" << attr.getValue().substr(1);
    } else if (auto const_op = llvm::dyn_cast<ScalarConstantOp>(op)) {
      if (auto attr = const_op.value().dyn_cast<IntegerAttr>()) {
        os << 'c' << attr.getValue();
      } else {
        os << "cst";
      }
    }
  }

  void getTypeAliases(mlir::SmallVectorImpl<std::pair<Type, StringRef>>& aliases) const final {
    auto ctx = getDialect()->getContext();
    for (const auto dataType : vertexai::tile::GetDataTypeSet()) {
      // Intern the string
      auto alias = mlir::Identifier::get(to_string(dataType), ctx);
      // Get the type
      auto type = getRankedTensorType(ScalarType::get(ctx, dataType));
      // Add the alias
      aliases.emplace_back(type, alias);
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

mlir::Type Dialect::parseType(mlir::DialectAsmParser& parser) const {  //
  return ScalarType::get(getContext(), DataTypeFromString(parser.getFullSymbolSpec()));
}

void Dialect::printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const {
  auto& os = printer.getStream();
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
  IVLOG(5, "eltwise::Dialect::materializeConstant> " << mlir::debugString(type));
  return builder.create<ScalarConstantOp>(loc, type, value);
}

static mlir::DialectRegistration<Dialect> EltwiseOps;

}  // namespace eltwise
}  // namespace dialect
}  // namespace pmlc
