// Copyright 2019, Intel Corporation

#pragma once

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

#include "pmlc/dialect/eltwise/types.h"
#include "pmlc/dialect/eltwise/util.h"
#include "pmlc/util/interfaces.h"

namespace pmlc {
namespace dialect {
namespace eltwise {

using llvm::SmallVector;
using mlir::AbstractOperation;
using mlir::APInt;
using mlir::ArrayRef;
using mlir::Attribute;
using mlir::Builder;
using mlir::FloatAttr;
using mlir::FloatType;
using mlir::IndexType;
using mlir::IntegerAttr;
using mlir::IntegerType;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::NamedAttribute;
using mlir::Op;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::OpFoldResult;
using mlir::OpInterface;
using mlir::OwningRewritePatternList;
using mlir::RankedTensorType;
using mlir::ShapedType;
using mlir::StringRef;
using mlir::TensorType;
using mlir::Type;
using mlir::TypeAttr;
using mlir::Value;
using mlir::VectorType;
using util::GenericBuilder;

namespace OpTrait = mlir::OpTrait;

#include "pmlc/dialect/eltwise/interfaces.h.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/eltwise/ops.h.inc"

}  // namespace eltwise
}  // namespace dialect
}  // namespace pmlc
