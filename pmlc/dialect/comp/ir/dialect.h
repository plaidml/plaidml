// Copyright 2020 Intel Corporation
#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "pmlc/dialect/comp/ir/interfaces.h"
#include "pmlc/dialect/comp/ir/types.h"

namespace pmlc::dialect::comp {
using mlir::ArrayRef;
using mlir::Attribute;
using mlir::DialectAsmParser;
using mlir::DialectAsmPrinter;
using mlir::DictionaryAttr;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MemoryEffectOpInterface;
using mlir::MemRefType;
using mlir::NamedAttribute;
using mlir::Op;
using mlir::OpAsmParser;
using mlir::OpAsmPrinter;
using mlir::OpAsmSetValueNameFn;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::OpFoldResult;
using mlir::OpInterface;
using mlir::ParseResult;
using mlir::Region;
using mlir::SmallVectorImpl;
using mlir::StringRef;
using mlir::Type;
using mlir::Value;
using mlir::ValueRange;

namespace OpTrait = mlir::OpTrait;
namespace SideEffects = mlir::SideEffects;
namespace MemoryEffects = mlir::MemoryEffects;

#define GET_OP_CLASSES
#include "pmlc/dialect/comp/ir/ops.h.inc"
#undef GET_OP_CLASSES

#include "pmlc/dialect/comp/ir/dialect.h.inc"

} // namespace pmlc::dialect::comp
