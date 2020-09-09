// Copyright 2020 Intel Corporation
#include "pmlc/dialect/comp/ir/types.h"

#include <string>

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;                // NOLINT
using namespace pmlc::dialect::comp; // NOLINT

bool RuntimeType::classof(Type type) {
  return type.isa<ExecEnvType, EventType>();
}

bool ExecEnvType::supportsMemorySpace(unsigned requestedSpace) const {
  bool memSpaceSupported = false;
  for (unsigned execEnvSpace : getMemorySpaces()) {
    memSpaceSupported |= execEnvSpace == requestedSpace;
  }
  return memSpaceSupported;
}

EventType ExecEnvType::getEventType() {
  return EventType::get(getContext(), *this);
}

static std::string runtimeToString(ExecEnvRuntime runtime) {
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

static ParseResult parseRuntime(DialectAsmParser &parser,
                                ExecEnvRuntime &runtime) {
  OptionalParseResult parseResult =
      parser.parseOptionalInteger(reinterpret_cast<unsigned &>(runtime));
  if (parseResult.hasValue())
    return parseResult.getValue();

  StringRef runtimeKeyword;
  if (failed(parser.parseKeyword(&runtimeKeyword)))
    return failure();

  auto failOrRuntime =
      llvm::StringSwitch<llvm::function_ref<FailureOr<ExecEnvRuntime>()>>(
          runtimeKeyword)
          .Case("vk", [] { return ExecEnvRuntime::Vulkan; })
          .Case("ocl", [] { return ExecEnvRuntime::OpenCL; })
          .Default([&] {
            auto loc = parser.getCurrentLocation();
            parser.emitError(loc, "unrecognized runtime string name: ")
                .append(runtimeKeyword)
                .attachNote()
                .append("available runtime names are: vk, ocl");
            return failure();
          })();

  if (failed(failOrRuntime))
    return failure();

  runtime = failOrRuntime.getValue();
  return success();
}

static void printEventType(EventType type, DialectAsmPrinter &printer) {
  printer << "event<" << runtimeToString(type.getRuntime()) << ">";
}

static Type parseEventType(DialectAsmParser &parser, Location loc) {
  ExecEnvRuntime runtime;
  if (parser.parseLess() || parseRuntime(parser, runtime) ||
      parser.parseGreater())
    return nullptr;

  return EventType::getChecked(runtime, loc);
}

static void printExecEnvType(ExecEnvType type, DialectAsmPrinter &printer) {
  printer << "execenv<";
  printer << runtimeToString(type.getRuntime());
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

static Type parseExecEnvType(DialectAsmParser &parser, Location loc) {
  ExecEnvRuntime runtime;
  unsigned tag;
  if (parser.parseLess() || parseRuntime(parser, runtime) ||
      parser.parseColon() || parser.parseInteger(tag) || parser.parseComma() ||
      parser.parseLParen())
    return nullptr;

  SmallVector<unsigned, 1> memorySpaces;
  bool first = true;
  while (true) {
    if (!first && parser.parseOptionalComma())
      break;
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

  return ExecEnvType::getChecked(runtime, tag, memorySpaces, loc);
}

void pmlc::dialect::comp::detail::printType(mlir::Type type,
                                            mlir::DialectAsmPrinter &printer) {
  llvm::TypeSwitch<mlir::Type>(type)
      .Case<EventType>(
          [&](EventType eventType) { printEventType(eventType, printer); })
      .Case<ExecEnvType>([&](ExecEnvType execEnvType) {
        printExecEnvType(execEnvType, printer);
      })
      .Default([](Type) { llvm_unreachable("Unhandled 'comp' type"); });
}

Type pmlc::dialect::comp::detail::parseType(mlir::DialectAsmParser &parser) {
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());

  StringRef typeKeyword;
  if (failed(parser.parseKeyword(&typeKeyword)))
    return nullptr;

  return llvm::StringSwitch<function_ref<Type()>>(typeKeyword)
      .Case("event", [&] { return parseEventType(parser, loc); })
      .Case("execenv", [&] { return parseExecEnvType(parser, loc); })
      .Default([&] {
        parser.emitError(parser.getNameLoc(),
                         "unknown comp type " + typeKeyword);
        return Type();
      })();
}
