// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace pmlc::dialect::xsmm {

using llvm::APInt;
using llvm::ArrayRef;
using llvm::StringRef;
using mlir::AffineMap;
using mlir::AffineMapAttr;
using mlir::ArrayAttr;
using mlir::Attribute;
using mlir::Block;
using mlir::Builder;
using mlir::DictionaryAttr;
using mlir::IndexType;
using mlir::IntegerAttr;
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
#include "pmlc/dialect/xsmm/ir/ops.h.inc"

#include "pmlc/dialect/xsmm/ir/dialect.h.inc"

} // namespace pmlc::dialect::xsmm
