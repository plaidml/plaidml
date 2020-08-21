// Copyright 2019, Intel Corporation

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
};

} // namespace

void EltwiseDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/eltwise/ir/ops.cc.inc"
      >();
  addInterfaces<OpAsmInterface>();
}

std::string EltwiseDialect::getCanonicalOpName(llvm::StringRef name) {
  if (name == "cond") {
    name = "select";
  }
  return llvm::formatv("{0}.{1}", getDialectNamespace(), name).str();
}

mlir::Operation *EltwiseDialect::materializeConstant(mlir::OpBuilder &builder,
                                                     mlir::Attribute value,
                                                     mlir::Type type,
                                                     mlir::Location loc) {
  IVLOG(5,
        "eltwise::Dialect::materializeConstant> " << mlir::debugString(type));
  return builder.create<ScalarConstantOp>(loc, type, value);
}

} // namespace pmlc::dialect::eltwise
