// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

#include "pmlc/dialect/scalar/types.h"

namespace pmlc {
namespace dialect {
namespace scalar {

using mlir::ArrayRef;
using mlir::Attribute;
using mlir::Builder;
using mlir::FloatAttr;
using mlir::FloatType;
using mlir::IndexType;
using mlir::IntegerAttr;
using mlir::IntegerType;
using mlir::LogicalResult;
using mlir::NamedAttribute;
using mlir::Op;
using mlir::Operation;
using mlir::OperationState;
using mlir::OpFoldResult;
using mlir::ShapedType;
using mlir::StringRef;
using mlir::TensorType;
using mlir::Type;
using mlir::TypeAttr;
using mlir::Value;
using mlir::VectorType;

namespace OpTrait = mlir::OpTrait;

#define GET_OP_CLASSES
#include "pmlc/dialect/scalar/ops.h.inc"

}  // namespace scalar
}  // namespace dialect
}  // namespace pmlc
