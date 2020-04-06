// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace pmlc::dialect::vulkan {

using llvm::ArrayRef;
using llvm::StringRef;
using mlir::Builder;
using mlir::LogicalResult;
using mlir::NamedAttribute;
using mlir::Op;
using mlir::Operation;
using mlir::OperationState;
using mlir::Type;
using mlir::Value;
using mlir::ValueRange;

namespace OpTrait = mlir::OpTrait;

#define GET_OP_CLASSES
#include "pmlc/dialect/vulkan/ir/ops.h.inc"

#include "pmlc/dialect/vulkan/ir/dialect.h.inc"

} // namespace pmlc::dialect::vulkan
