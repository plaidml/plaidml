// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

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
using mlir::IndexType;
using mlir::IntegerAttr;
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
using mlir::Region;
using mlir::Type;
using mlir::Value;
using mlir::ValueRange;

namespace OpTrait = mlir::OpTrait;

#define GET_OP_CLASSES
#include "pmlc/dialect/xsmm/ir/ops.h.inc"

} // namespace pmlc::dialect::xsmm
