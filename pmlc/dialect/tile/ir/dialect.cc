// Copyright 2019, Intel Corporation

#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::tile {

namespace {

struct OpAsmDialectInterfaceImpl : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  // Get a special name to use when printing the given operation.
  void getAsmResultNames(Operation *op,
                         OpAsmSetValueNameFn setNameFn) const final {
    llvm::SmallString<32> osbuf;
    llvm::raw_svector_ostream os(osbuf);
    if (auto constOp = llvm::dyn_cast<ConstantOp>(op)) {
      os << 'c' << constOp.value().getSExtValue();
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
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/tile/ir/ops.cc.inc"
      >();
  addInterfaces<OpAsmDialectInterfaceImpl>();
}

std::string TileDialect::getDialectAttrName(StringRef name) {
  return llvm::formatv("{0}.{1}", getDialectNamespace(), name).str();
}

std::string TileDialect::getCanonicalOpName(StringRef name) {
  return llvm::formatv("{0}.{1}", getDialectNamespace(), name).str();
}

Operation *TileDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  IVLOG(5, "tile::TileDialect::materializeConstant> "
               << debugString(value) << " : " << debugString(type));
  if (auto attr = value.dyn_cast<IntegerAttr>()) {
    auto indexType = builder.getIndexType();
    auto indexAttr = builder.getIntegerAttr(indexType, attr.getInt());
    return builder.create<ConstantOp>(loc, indexType, indexAttr);
  }
  return nullptr;
}

} // namespace pmlc::dialect::tile
