// Copyright 2019, Intel Corporation

#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/util/logging.h"

namespace pmlc::dialect::tile {

namespace {

struct OpAsmInterface : public mlir::OpAsmDialectInterface {
  using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

  // Get a special name to use when printing the given operation.
  void getAsmResultNames(Operation *op,
                         mlir::OpAsmSetValueNameFn setNameFn) const final {
    llvm::SmallString<32> osbuf;
    llvm::raw_svector_ostream os(osbuf);
    if (auto constOp = llvm::dyn_cast<ConstantOp>(op)) {
      os << 'c' << constOp.value().getSExtValue();
    } else if (auto indexOp = llvm::dyn_cast<PolyIndexOp>(op)) {
      if (indexOp.name().hasValue()) {
        os << *indexOp.name();
      }
    } else if (auto cionOp = llvm::dyn_cast<SymbolicContractionOp>(op)) {
      if (cionOp.name().hasValue()) {
        os << *cionOp.name();
      }
    } else if (auto cionOp = llvm::dyn_cast<ContractionOp>(op)) {
      if (cionOp.name().hasValue()) {
        os << *cionOp.name();
      }
    }
    setNameFn(op->getResult(0), os.str());
  }
};

} // namespace

void TileDialect::initialize() {
  addTypes<AffineMapType, AffineConstraintsType, AffineTensorMapType,
           StringType>();
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/tile/ir/ops.cc.inc"
      >();
  addInterfaces<OpAsmInterface>();
}

std::string TileDialect::getDialectAttrName(StringRef name) {
  return llvm::formatv("{0}.{1}", getDialectNamespace(), name).str();
}

std::string TileDialect::getCanonicalOpName(StringRef name) {
  return llvm::formatv("{0}.{1}", getDialectNamespace(), name).str();
}

void TileDialect::printType(Type type, mlir::DialectAsmPrinter &printer) const {
  auto &os = printer.getStream();
  if (type.isa<AffineTensorMapType>()) {
    os << "tmap";
  } else if (type.isa<AffineMapType>()) {
    os << "map";
  } else if (type.isa<AffineConstraintsType>()) {
    os << "cons";
  }
}

Type TileDialect::parseType(mlir::DialectAsmParser &parser) const {
  auto spec = parser.getFullSymbolSpec();
  auto type = llvm::StringSwitch<Type>(spec)
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

Operation *TileDialect::materializeConstant(mlir::OpBuilder &builder,
                                            Attribute value, Type type,
                                            Location loc) {
  IVLOG(5, "tile::TileDialect::materializeConstant> "
               << mlir::debugString(value) << " : " << mlir::debugString(type));
  if (auto attr = value.dyn_cast<IntegerAttr>()) {
    auto indexType = builder.getIndexType();
    auto indexAttr = builder.getIntegerAttr(indexType, attr.getInt());
    return builder.create<ConstantOp>(loc, indexType, indexAttr);
  }
  return nullptr;
}

} // namespace pmlc::dialect::tile
