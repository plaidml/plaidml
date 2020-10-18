// Copyright 2019, Intel Corporation

#include "pmlc/dialect/abi/ir/dialect.h"

// #include "mlir/Dialect/StandardOps/IR/Ops.h"
// #include "mlir/IR/OpImplementation.h"

namespace pmlc::dialect::abi {

using namespace mlir; // NOLINT

#define GET_OP_CLASSES
#include "pmlc/dialect/abi/ir/ops.cc.inc"

void ABIDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/abi/ir/ops.cc.inc" // NOLINT
      >();
}

mlir::Region &LoopOp::getLoopBody() { return region(); }

bool LoopOp::isDefinedOutsideOfLoop(mlir::Value value) {
  return !region().isAncestor(value.getParentRegion());
}

LogicalResult LoopOp::moveOutOfLoop(mlir::ArrayRef<mlir::Operation *> ops) {
  for (auto op : ops) {
    op->moveBefore(*this);
  }
  return mlir::success();
}

} // namespace pmlc::dialect::abi
