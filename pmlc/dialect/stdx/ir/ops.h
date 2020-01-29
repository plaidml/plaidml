// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

namespace pmlc::dialect::stdx {

using llvm::ArrayRef;
using llvm::StringRef;
using mlir::Block;
using mlir::Builder;
using mlir::IndexType;
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
#include "pmlc/dialect/stdx/ir/ops.h.inc"

}  // namespace pmlc::dialect::stdx
