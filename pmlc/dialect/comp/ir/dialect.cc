// Copyright 2020 Intel Corporation

#include "pmlc/dialect/comp/ir/dialect.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::comp {

// ============================================================================
// Resources
// ============================================================================

struct ExecEnvResource final
    : public mlir::SideEffects::Resource::Base<ExecEnvResource> {
  StringRef getName() final { return "ExecEnv"; }
};

struct DeviceMemoryResource final
    : public mlir::SideEffects::Resource::Base<DeviceMemoryResource> {
  StringRef getName() final { return "DeviceMemory"; }
};

struct KernelResource final
    : public mlir::SideEffects::Resource::Base<KernelResource> {
  StringRef getName() final { return "Kernel"; }
};

// ============================================================================
// Operations
// ============================================================================

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

void COMPDialect::initialize() {
  addTypes<DeviceType, ExecEnvType, EventType, KernelType>();
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

} // namespace pmlc::dialect::comp

#define GET_OP_CLASSES
#include "pmlc/dialect/comp/ir/ops.cc.inc" // NOLINT
