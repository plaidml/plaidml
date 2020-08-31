// Copyright 2020 Intel Corporation

#include "pmlc/dialect/stdx/ir/ops.h"

#include "mlir/IR/OpImplementation.h"

namespace pmlc::dialect::stdx {

using mlir::failure;
using mlir::success;

// ---- ReshapeOp ----

Value ReshapeOp::getViewSource() { return tensor(); }

// ---- SubgroupBlockReadINTELOp ----

static LogicalResult
verifySubgroupBlockReadINTELOp(SubgroupBlockReadINTELOp op) {
  if (op.getNumOperands() != 1 + op.getMemRefType().getRank())
    return op.emitOpError("incorrect number of indices for memory ref");
  return success();
}

// ---- SubgroupBlockWriteINTELOp ----

static LogicalResult
verifySubgroupBlockWriteINTELOp(SubgroupBlockWriteINTELOp op) {
  if (op.getNumOperands() != 2 + op.getMemRefType().getRank())
    return op.emitOpError("subgroup block write index operand"
                          " count not equal to memref rank");

  return success();
}

#define GET_OP_CLASSES
#include "pmlc/dialect/stdx/ir/ops.cc.inc"

void StdXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/stdx/ir/ops.cc.inc" // NOLINT
      >();
}

} // namespace pmlc::dialect::stdx
