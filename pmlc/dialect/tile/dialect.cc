// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/dialect.h"

#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "base/util/logging.h"
#include "pmlc/dialect/tile/ops.h"

namespace pmlc {
namespace dialect {
namespace tile {

namespace {

struct OpAsmInterface : public mlir::OpAsmDialectInterface {
  using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

  /// Get a special name to use when printing the given operation. The desired
  /// name should be streamed into 'os'.
  void getOpResultName(Operation* op, llvm::raw_ostream& os) const final {
    if (auto constOp = llvm::dyn_cast<AffineConstantOp>(op)) {
      os << 'c' << constOp.value().getSExtValue();
    }
  }
};

}  // namespace

Dialect::Dialect(mlir::MLIRContext* ctx) : mlir::Dialect(getDialectNamespace(), ctx) {
  addTypes<AffineIndexMapType, AffineSizeMapType, StringType>();
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/tile/ops.cc.inc"
      >();
  addInterfaces<OpAsmInterface>();
}

std::string Dialect::getDialectAttrName(llvm::StringRef name) {
  return llvm::formatv("{0}.{1}", getDialectNamespace(), name).str();
}

std::string Dialect::getCanonicalOpName(llvm::StringRef name) {
  return llvm::formatv("{0}.{1}", getDialectNamespace(), name).str();
}

void Dialect::printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const {
  auto& os = printer.getStream();
  if (auto t = type.dyn_cast<AffineIndexMapType>()) {
    os << "imap";
  } else if (auto t = type.dyn_cast<AffineSizeMapType>()) {
    os << "smap";
  }
}

mlir::Type Dialect::parseType(mlir::DialectAsmParser& parser) const {
  StringRef spec = parser.getFullSymbolSpec();
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  if (spec == "imap") {
    return AffineIndexMapType::get(getContext());
  }
  if (spec == "smap") {
    return AffineSizeMapType::get(getContext());
  }
  emitError(loc, llvm::formatv("unknown tile type: '{0}'", spec));
  return Type();
}

mlir::Operation* Dialect::materializeConstant(  //
    mlir::OpBuilder& builder,                   //
    mlir::Attribute value,                      //
    mlir::Type type,                            //
    mlir::Location loc) {
  IVLOG(5, "tile::Dialect::materializeConstant");
  if (auto attr = value.dyn_cast<IntegerAttr>()) {
    return builder.create<AffineConstantOp>(loc, type, attr);
  }
  return nullptr;
}

AffineIndexMapType AffineIndexMapType::get(mlir::MLIRContext* context) {  //
  return Base::get(context, Kinds::AffineIndexMap);
}

AffineSizeMapType AffineSizeMapType::get(mlir::MLIRContext* context) {  //
  return Base::get(context, Kinds::AffineSizeMap);
}

StringType StringType::get(mlir::MLIRContext* context) {  //
  return Base::get(context, Kinds::String);
}

static mlir::DialectRegistration<Dialect> EdslOps;

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
