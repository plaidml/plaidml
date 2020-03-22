// Copyright 2019, Intel Corporation

#include "pmlc/dialect/eltwise/ir/dialect.h"

#include <utility>

#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/util/logging.h"

#define DEBUG_TYPE "eltwise"

namespace pmlc::dialect::eltwise {

namespace {

struct OpAsmInterface : public mlir::OpAsmDialectInterface {
  using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

  /// Get a special name to use when printing the given operation.
  void getAsmResultNames(Operation *op,
                         mlir::OpAsmSetValueNameFn setNameFn) const final {
    llvm::SmallString<32> osbuf;
    llvm::raw_svector_ostream os(osbuf);
    if (auto attr = op->getAttrOfType<mlir::StringAttr>("scalar_name")) {
      os << "s_" << attr.getValue().substr(1);
    } else if (auto const_op = llvm::dyn_cast<ScalarConstantOp>(op)) {
      if (auto attr = const_op.value().dyn_cast<IntegerAttr>()) {
        os << 'c' << attr.getValue();
      } else {
        os << "cst";
      }
    }
    setNameFn(op->getResult(0), os.str());
  }

  // void getTypeAliases(
  //     mlir::SmallVectorImpl<std::pair<Type, StringRef>> &aliases) const final
  //     {
  //   auto ctx = getDialect()->getContext();
  //   for (uint64_t i = 1, e = util::getMaxEnumValForDataType(); i <= e; ++i) {
  //     auto dtype = util::symbolizeDataType(i).getValue();
  //     auto dtypeStr = util::stringifyDataType(dtype);
  //     // Intern the string
  //     auto alias = mlir::Identifier::get(dtypeStr, ctx);
  //     // Get the type
  //     auto type = getRankedTensorType(ScalarType::get(ctx, dtype));
  //     // Add the alias
  //     aliases.emplace_back(type, alias);
  //   }
  // }
};

} // namespace

Dialect::Dialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx) {
  // addTypes<ScalarType>();
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/eltwise/ir/ops.cc.inc"
      >();
  addInterfaces<OpAsmInterface>();
}

std::string Dialect::getCanonicalOpName(llvm::StringRef name) {
  if (name == "cond") {
    name = "select";
  }
  return llvm::formatv("{0}.{1}", getDialectNamespace(), name).str();
}

// mlir::Type Dialect::parseType(mlir::DialectAsmParser &parser) const {
//   auto dtype = util::symbolizeDataType(parser.getFullSymbolSpec());
//   if (!dtype.hasValue()) {
//     parser.emitError(parser.getNameLoc(), "unknown eltwise type: ")
//         << parser.getFullSymbolSpec();
//     return {};
//   }
//   return ScalarType::get(getContext(), dtype.getValue());
// }

// void Dialect::printType(mlir::Type type,
//                         mlir::DialectAsmPrinter &printer) const {
//   auto &os = printer.getStream();
//   if (auto scalarType = type.dyn_cast<ScalarType>()) {
//     os << util::stringifyDataType(scalarType.type());
//   } else {
//     llvm_unreachable("unhandled scalar type");
//   }
// }

mlir::Operation *Dialect::materializeConstant(mlir::OpBuilder &builder,
                                              mlir::Attribute value,
                                              mlir::Type type,
                                              mlir::Location loc) {
  IVLOG(5,
        "eltwise::Dialect::materializeConstant> " << mlir::debugString(type));
  return builder.create<ScalarConstantOp>(loc, type, value);
}

} // namespace pmlc::dialect::eltwise
