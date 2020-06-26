// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "pmlc/dialect/vulkan/ir/types.h"

namespace pmlc::dialect::vulkan {

using llvm::ArrayRef;
using llvm::StringRef;
using mlir::Builder;
using mlir::DictionaryAttr;
using mlir::FlatSymbolRefAttr;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MemRefType;
using mlir::NamedAttribute;
using mlir::Op;
using mlir::OpAsmParser;
using mlir::OpAsmPrinter;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::ParseResult;
using mlir::StringAttr;
using mlir::Type;
using mlir::Value;
using mlir::ValueRange;

namespace OpTrait = mlir::OpTrait;

#define GET_OP_CLASSES
#include "pmlc/dialect/vulkan/ir/ops.h.inc"

#include "pmlc/dialect/vulkan/ir/dialect.h.inc"

} // namespace pmlc::dialect::vulkan
