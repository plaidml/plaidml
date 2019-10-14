// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/dialect.h"

#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

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
    if (auto const_op = llvm::dyn_cast<AffineConstantOp>(op)) {
      auto value = const_op.value().getSExtValue();
      os << 'c' << value;
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

std::string Dialect::getCanonicalOpName(llvm::StringRef name) {
  return llvm::formatv("{0}.{1}", getDialectNamespace(), name).str();
}

void Dialect::printType(mlir::Type type, llvm::raw_ostream& os) const {
  if (auto t = type.dyn_cast<AffineIndexMapType>()) {
    os << "imap";
  } else if (auto t = type.dyn_cast<AffineSizeMapType>()) {
    os << "smap";
  }
}

mlir::Operation* Dialect::materializeConstant(  //
    mlir::OpBuilder& builder,                   //
    mlir::Attribute value,                      //
    mlir::Type type,                            //
    mlir::Location loc) {
  auto int_attr = value.dyn_cast<IntegerAttr>();
  if (int_attr) {
    return builder.create<AffineConstantOp>(loc, type, int_attr);
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
