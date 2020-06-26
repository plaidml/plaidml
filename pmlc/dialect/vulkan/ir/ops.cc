// Copyright 2020 Intel Corporation

#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "pmlc/dialect/vulkan/ir/ops.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::vulkan {

#define GET_OP_CLASSES
#include "pmlc/dialect/vulkan/ir/ops.cc.inc"

VkDialect::VkDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<BufferType, ShaderModuleType>();
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/vulkan/ir/ops.cc.inc" // NOLINT
      >();
}

void VkDialect::printType(Type type, mlir::DialectAsmPrinter &printer) const {
  auto &os = printer.getStream();
  if (type.isa<BufferType>()) {
    os << "vbuffer";
  } else if (type.isa<ShaderModuleType>()) {
    os << "ShaderModule";
  }
}

Type VkDialect::parseType(mlir::DialectAsmParser &parser) const {
  auto spec = parser.getFullSymbolSpec();
  auto type = llvm::StringSwitch<Type>(spec)
                  .Case("vbuffer", BufferType::get(getContext()))
                  .Case("ShaderModule", ShaderModuleType::get(getContext()))
                  .Default(Type());
  if (!type) {
    auto loc = parser.getEncodedSourceLoc(parser.getNameLoc());
    emitError(loc, llvm::formatv("Unknown vulkan type: '{0}'", spec));
  }
  return type;
}

} // namespace pmlc::dialect::vulkan
