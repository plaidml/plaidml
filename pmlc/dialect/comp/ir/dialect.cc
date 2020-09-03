// Copyright 2020 Intel Corporation
#include "pmlc/dialect/comp/ir/dialect.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;                // NOLINT
using namespace pmlc::dialect::comp; // NOLINT

// ============================================================================
// Operations
// ============================================================================
// Verifiers
static LogicalResult verify(ScheduleFunc op) {
  static const char *errorMsg =
      "body must have one operation - 'gpu.launch_func'";

  Block &region = op.body().front();
  auto opsRange = region.without_terminator();
  if (opsRange.empty())
    return op.emitOpError(errorMsg);
  if (++opsRange.begin() != opsRange.end())
    return op.emitOpError(errorMsg);

  Operation &onlyOp = *opsRange.begin();
  if (!isa<gpu::LaunchFuncOp>(onlyOp))
    return op.emitOpError(errorMsg);

  return success();
}

static LogicalResult verify(Alloc op) {
  Type deviceMemType = op.deviceMem().getType();

  if (op.hostMem()) {
    auto hostMemType = op.hostMem().getType();
    if (deviceMemType.cast<::mlir::ShapedType>().getShape() !=
        hostMemType.cast<::mlir::ShapedType>().getShape())
      return op.emitOpError("host and device memory shapes must match");
    if (getElementTypeOrSelf(deviceMemType) !=
        getElementTypeOrSelf(hostMemType))
      return op.emitOpError("host and device memory element types must match");
  }

  return success();
}

// Interface implementations
::mlir::Value ScheduleWrite::getSource() { return hostMem(); }
::mlir::Value ScheduleWrite::getDestination() { return deviceMem(); }
::mlir::Value ScheduleWrite::getSourceExecEnv() { return mlir::Value(); }
::mlir::Value ScheduleWrite::getDestinationExecEnv() { return execEnv(); }

::mlir::Value ScheduleRead::getSource() { return deviceMem(); }
::mlir::Value ScheduleRead::getDestination() { return hostMem(); }
::mlir::Value ScheduleRead::getSourceExecEnv() { return execEnv(); }
::mlir::Value ScheduleRead::getDestinationExecEnv() { return mlir::Value(); }

// ============================================================================
// Dialect
// ============================================================================

#define GET_OP_CLASSES
#include "pmlc/dialect/comp/ir/ops.cc.inc"
#undef GET_OP_CLASSES

void COMPDialect::initialize() {
  addTypes<ExecEnvType, EventType>();
#define GET_OP_LIST
  addOperations<
#include "pmlc/dialect/comp/ir/ops.cc.inc" // NOLINT
      >();
#undef GET_OP_LIST
}

void COMPDialect::printType(Type type, DialectAsmPrinter &printer) const {
  return detail::printType(type, printer);
}

Type COMPDialect::parseType(DialectAsmParser &parser) const {
  return detail::parseType(parser);
}
