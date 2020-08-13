// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace pmlc::dialect::stdx {

using llvm::ArrayRef;
using llvm::StringRef;
using mlir::Block;
using mlir::Builder;
using mlir::DictionaryAttr;
using mlir::IndexType;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MemoryEffectOpInterface;
using mlir::MemRefType;
using mlir::NamedAttribute;
using mlir::Op;
using mlir::OpAsmParser;
using mlir::OpAsmPrinter;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::ParseResult;
using mlir::Region;
using mlir::SmallVectorImpl;
using mlir::Type;
using mlir::Value;
using mlir::ValueRange;

namespace MemoryEffects = mlir::MemoryEffects;
namespace OpTrait = mlir::OpTrait;
namespace SideEffects = mlir::SideEffects;

#define GET_OP_CLASSES
#include "pmlc/dialect/stdx/ir/ops.h.inc"

#include "pmlc/dialect/stdx/ir/dialect.h.inc"

} // namespace pmlc::dialect::stdx
