// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

#include "pmlc/dialect/eltwise/types.h"
#include "pmlc/dialect/tile/types.h"

namespace pmlc {
namespace dialect {
namespace tile {

using eltwise::ScalarType;
using llvm::APInt;
using mlir::ArrayRef;
using mlir::Attribute;
using mlir::Builder;
using mlir::IndexType;
using mlir::IntegerAttr;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::NamedAttribute;
using mlir::NoneType;
using mlir::Op;
using mlir::Operation;
using mlir::OperationState;
using mlir::OpFoldResult;
using mlir::OpInterface;
using mlir::OwningRewritePatternList;
using mlir::RankedTensorType;
using mlir::Region;
using mlir::ShapedType;
using mlir::StringAttr;
using mlir::StringRef;
using mlir::TupleType;
using mlir::Type;
using mlir::TypeAttr;
using mlir::Value;

namespace OpTrait = mlir::OpTrait;

#include "pmlc/dialect/tile/opinterfaces.h.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/tile/ops.h.inc"

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
