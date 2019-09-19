// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/OpDefinition.h"
#include "pmlc/dialect/stripe/types.h"

namespace pmlc {
namespace dialect {
namespace stripe {

using eltwise::ScalarType;
using mlir::ArrayRef;
using mlir::Builder;
using mlir::IndexType;
using mlir::LogicalResult;
using mlir::NamedAttribute;
using mlir::Op;
using mlir::Operation;
using mlir::OperationState;
using mlir::OpFoldResult;
using mlir::RankedTensorType;
using mlir::ShapedType;
using mlir::StringRef;
using mlir::Type;
using mlir::Value;

namespace OpTrait = mlir::OpTrait;

#define GET_OP_CLASSES
#include "pmlc/dialect/stripe/ops.h.inc"

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc

#include "llvm/ADT/DenseMapInfo.h"
using llvm::DenseMapInfo;
#include "pmlc/dialect/stripe/ops_enum.h.inc"
