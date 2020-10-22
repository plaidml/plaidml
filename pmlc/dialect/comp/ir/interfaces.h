// Copyright 2020 Intel Corporation
#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/InterfaceSupport.h"
#include "mlir/Support/LogicalResult.h"

namespace pmlc::dialect::comp {

mlir::LogicalResult verifyMemoryTransferOp(mlir::Operation *op);
mlir::LogicalResult verifyScheduleOp(mlir::Operation *op);

#include "pmlc/dialect/comp/ir/interfaces.h.inc"

} // namespace pmlc::dialect::comp
