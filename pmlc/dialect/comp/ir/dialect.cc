#include "pmlc/dialect/comp/ir/dialect.h"

#include <string>

#include "mlir/IR/DialectImplementation.h"

namespace pmlc::dialect::comp {

using namespace mlir; // NOLINT

#include "pmlc/dialect/comp/ir/interfaces.cc.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/comp/ir/ops.cc.inc"
#undef GET_OP_CLASSES

COMPDialect::COMPDialect(::mlir::MLIRContext *context)
    : mlir::Dialect(getDialectNamespace(), context) {
  addTypes<ExecEnvType, EventType>();
#define GET_OP_LIST
  addOperations<
#include "pmlc/dialect/comp/ir/ops.cc.inc" // NOLINT
      >();
#undef GET_OP_LIST
}

// ============================================================================
// Type parsing and prinitng
// ============================================================================
namespace {

std::string execEnvToString(ExecEnvRuntime runtime) {
  switch (runtime) {
  case ExecEnvRuntime::Vulkan:
    return "vk";
  case ExecEnvRuntime::OpenCL:
    return "ocl";
  default: {
    unsigned customNr = static_cast<unsigned>(runtime);
    return std::to_string(customNr);
  }
  }
}

ExecEnvRuntime execEnvFromString(StringRef str) {
  if (str == "vk")
    return ExecEnvRuntime::Vulkan;
  if (str == "ocl")
    return ExecEnvRuntime::OpenCL;
  unsigned customNr;
  str.getAsInteger(10, customNr);
  return static_cast<ExecEnvRuntime>(customNr);
}

void printEventType(EventType type, DialectAsmPrinter &printer) {
  printer << "event<" << execEnvToString(type.getRuntime()) << ">";
}

Type parseEventType(DialectAsmParser &parser, Location loc) {
  if (parser.parseLess())
    return nullptr;

  StringRef runtimeKeyword;
  if (failed(parser.parseKeyword(&runtimeKeyword)))
    return nullptr;

  ExecEnvRuntime runtime = execEnvFromString(runtimeKeyword);

  if (parser.parseGreater())
    return nullptr;

  return EventType::getChecked(runtime, loc);
}

void printExecEnvType(ExecEnvType type, DialectAsmPrinter &printer) {
  printer << "execenv<";
  printer << execEnvToString(type.getRuntime());
  printer << ":" << type.getTag() << ",(";
  bool first = true;
  for (auto &memSpace : type.getMemorySpaces()) {
    if (!first)
      printer << ",";
    printer << memSpace;
    first = false;
  }
  printer << ")>";
}

Type parseExecEnvType(DialectAsmParser &parser, Location loc) {
  StringRef runtimeKeyword;
  unsigned tag;
  if (parser.parseLess() || parser.parseKeyword(&runtimeKeyword) ||
      parser.parseColon() || parser.parseInteger(tag) || parser.parseComma() ||
      parser.parseLParen())
    return nullptr;

  SmallVector<unsigned, 1> memorySpaces;
  bool first = true;
  while (true) {
    if (!first) {
      if (parser.parseOptionalComma())
        break;
    }
    unsigned memSpace;
    auto optionalInt = parser.parseOptionalInteger(memSpace);
    if (optionalInt.hasValue())
      memorySpaces.push_back(memSpace);
    else
      break;
    first = false;
  }

  if (parser.parseRParen() || parser.parseGreater())
    return nullptr;

  ExecEnvRuntime runtime = execEnvFromString(runtimeKeyword);
  return ExecEnvType::getChecked(runtime, tag, memorySpaces, loc);
}

} // namespace

void COMPDialect::printType(Type type, DialectAsmPrinter &printer) const {
  switch (type.getKind()) {
  case CompTypes::Event:
    printEventType(type.cast<EventType>(), printer);
    break;
  case CompTypes::ExecEnv:
    printExecEnvType(type.cast<ExecEnvType>(), printer);
    break;
  default:
    llvm_unreachable("Unhandled comp type");
  }
}

Type COMPDialect::parseType(DialectAsmParser &parser) const {
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());

  StringRef typeKeyword;
  if (failed(parser.parseKeyword(&typeKeyword)))
    return nullptr;

  if (typeKeyword == "event")
    return parseEventType(parser, loc);
  if (typeKeyword == "execenv")
    return parseExecEnvType(parser, loc);

  parser.emitError(parser.getNameLoc(), "unknown comp type " + typeKeyword);

  return nullptr;
}

// ============================================================================
// Operations
// ============================================================================
::mlir::Value ScheduleRead::getSource() { return deviceMem(); }
::mlir::Value ScheduleRead::getDestination() { return hostMem(); }
::mlir::Value ScheduleRead::getSourceExecEnv() { return execEnv(); }
::mlir::Value ScheduleRead::getDestinationExecEnv() { return mlir::Value(); }

::mlir::Value ScheduleWrite::getSource() { return hostMem(); }
::mlir::Value ScheduleWrite::getDestination() { return deviceMem(); }
::mlir::Value ScheduleWrite::getSourceExecEnv() { return mlir::Value(); }
::mlir::Value ScheduleWrite::getDestinationExecEnv() { return execEnv(); }

::mlir::Value ScheduleCopy::getSource() { return srcMem(); }
::mlir::Value ScheduleCopy::getDestination() { return dstMem(); }
::mlir::Value ScheduleCopy::getSourceExecEnv() { return execEnv(); }
::mlir::Value ScheduleCopy::getDestinationExecEnv() { return execEnv(); }

OpFoldResult GroupEvents::fold(ArrayRef<Attribute> operands) {
  // If just one event to group, pass it on
  if (operands.size() == 1)
    return events()[0];

  return {};
}

} // namespace pmlc::dialect::comp
