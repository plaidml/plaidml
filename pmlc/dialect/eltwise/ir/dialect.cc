// Copyright 2019, Intel Corporation

#include <utility>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/eltwise/ir/types.h"
#include "pmlc/util/logging.h"

#define DEBUG_TYPE "eltwise"

using namespace mlir; // NOLINT

namespace pmlc::dialect::eltwise {

namespace {

struct OpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  /// Get a special name to use when printing the given operation.
  void getAsmResultNames(Operation *op,
                         OpAsmSetValueNameFn setNameFn) const final {
    llvm::SmallString<32> osbuf;
    llvm::raw_svector_ostream os(osbuf);
    if (auto attr = op->getAttrOfType<StringAttr>("scalar_name")) {
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
};

} // namespace

void EltwiseDialect::initialize() {
  addTypes<APFloatType, APSignedIntegerType, APUnsignedIntegerType>();
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/eltwise/ir/ops.cc.inc"
      >();
  addInterfaces<OpAsmInterface>();
}

std::string EltwiseDialect::getCanonicalOpName(llvm::StringRef name) {
  return llvm::formatv("{0}.{1}", getDialectNamespace(), name).str();
}

Operation *EltwiseDialect::materializeConstant(OpBuilder &builder,
                                               Attribute value, Type type,
                                               Location loc) {
  IVLOG(3, "eltwise::Dialect::materializeConstant> "
               << debugString(value) << " : " << debugString(type));
  auto rankedTensorType = getRankedTensorType(type);
  Type elementType = rankedTensorType.getElementType();
  if (elementType.isa<FloatType>()) {
    if (auto attr = value.dyn_cast<IntegerAttr>()) {
      return builder.create<ScalarConstantOp>(
          loc, elementType, static_cast<double>(attr.getInt()));
    }
  }
  if (elementType.isa<IntegerType>()) {
    if (auto attr = value.dyn_cast<FloatAttr>()) {
      return builder.create<ScalarConstantOp>(
          loc, elementType, static_cast<int64_t>(attr.getValueAsDouble()));
    }
  }
  return builder.create<ScalarConstantOp>(loc, type, value);
}

void EltwiseDialect::printType(Type type, DialectAsmPrinter &printer) const {
  llvm::TypeSwitch<Type>(type)
      .Case<APFloatType>([&](auto deviceType) { printer << "fx"; })
      .Case<APSignedIntegerType>([&](auto eventType) { printer << "six"; })
      .Case<APUnsignedIntegerType>([&](auto eventType) { printer << "uix"; })
      .Default([](Type) { llvm_unreachable("Unsupported 'eltwise' type"); });
}

Type EltwiseDialect::parseType(DialectAsmParser &parser) const {
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());

  StringRef typeKeyword;
  if (failed(parser.parseKeyword(&typeKeyword)))
    return nullptr;

  return llvm::StringSwitch<function_ref<Type()>>(typeKeyword)
      .Case("fx", [&] { return APFloatType::getChecked(loc); })
      .Case("six", [&] { return APSignedIntegerType::getChecked(loc); })
      .Case("uix", [&] { return APUnsignedIntegerType::getChecked(loc); })
      .Default([&] {
        parser.emitError(parser.getNameLoc(),
                         "Unsupported 'eltwise' type: " + typeKeyword);
        return Type();
      })();
}

} // namespace pmlc::dialect::eltwise
