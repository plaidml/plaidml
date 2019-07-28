// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/OpDefinition.h"

namespace pmlc {
namespace dialect {
namespace hir {

using mlir::ArrayRef;
using mlir::Builder;
using mlir::LogicalResult;
using mlir::NamedAttribute;
using mlir::Op;
using mlir::Operation;
using mlir::OperationState;
using mlir::StringRef;
using mlir::Type;
using mlir::Value;

namespace OpTrait = mlir::OpTrait;

#define GET_OP_CLASSES
#include "pmlc/dialect/hir/ops.h.inc"

}  // namespace hir
}  // namespace dialect
}  // namespace pmlc
