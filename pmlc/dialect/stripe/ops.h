// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/OpDefinition.h"

#include "pmlc/dialect/eltwise/ops.h"
#include "pmlc/dialect/stripe/types.h"

namespace pmlc {
namespace dialect {
namespace stripe {

using eltwise::AggregationKind;
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
using mlir::OpInterface;
using mlir::RankedTensorType;
using mlir::ShapedType;
using mlir::StringRef;
using mlir::Type;
using mlir::Value;

namespace OpTrait = mlir::OpTrait;

#include "pmlc/dialect/stripe/ops_interfaces.h.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/stripe/ops.h.inc"

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
