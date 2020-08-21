// Copyright 2020 Intel Corporation

#include "pmlc/dialect/stdx/ir/ops.h"

#include "mlir/IR/OpImplementation.h"

namespace pmlc::dialect::stdx {

using mlir::failure;
using mlir::success;

// ---- ReshapeOp ----

Value ReshapeOp::getViewSource() { return tensor(); }

#define GET_OP_CLASSES
#include "pmlc/dialect/stdx/ir/ops.cc.inc"

void StdXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/stdx/ir/ops.cc.inc" // NOLINT
      >();
}

} // namespace pmlc::dialect::stdx
