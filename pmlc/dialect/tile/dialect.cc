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

  // Get a special name to use when printing the given operation.
  void getAsmResultNames(Operation* op, mlir::OpAsmSetValueNameFn setNameFn) const final {
    llvm::SmallString<32> osbuf;
    llvm::raw_svector_ostream os(osbuf);
    if (auto constOp = llvm::dyn_cast<AffineConstantOp>(op)) {
      os << 'c' << constOp.value().getSExtValue();
    }
    setNameFn(op->getResult(0), os.str());
  }
};

}  // namespace

Dialect::Dialect(mlir::MLIRContext* ctx) : mlir::Dialect(getDialectNamespace(), ctx) {
  addTypes<                   //
      AffineMapType,          //
      AffineConstraintsType,  //
      AffineIndexMapType,     //
      AffineSizeMapType,      //
      AffineTensorMapType,    //
      StringType>();
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
  if (type.isa<AffineIndexMapType>()) {
    os << "imap";
  } else if (type.isa<AffineTensorMapType>()) {
    os << "tmap";
  } else if (type.isa<AffineMapType>()) {
    os << "map";
  } else if (type.isa<AffineConstraintsType>()) {
    os << "cons";
  } else if (type.isa<AffineSizeMapType>()) {
    os << "smap";
  }
}

mlir::Type Dialect::parseType(mlir::DialectAsmParser& parser) const {
  auto spec = parser.getFullSymbolSpec();
  auto type = llvm::StringSwitch<mlir::Type>(spec)
                  .Case("imap", AffineIndexMapType::get(getContext()))
                  .Case("smap", AffineSizeMapType::get(getContext()))
                  .Case("tmap", AffineTensorMapType::get(getContext()))
                  .Case("map", AffineMapType::get(getContext()))
                  .Case("cons", AffineConstraintsType::get(getContext()))
                  .Default(Type());
  if (!type) {
    auto loc = parser.getEncodedSourceLoc(parser.getNameLoc());
    emitError(loc, llvm::formatv("unknown tile type: '{0}'", spec));
  }
  return type;
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

static mlir::DialectRegistration<Dialect> EdslOps;

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
