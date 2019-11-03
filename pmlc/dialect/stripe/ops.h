// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/OpDefinition.h"

#include "pmlc/dialect/eltwise/types.h"
#include "pmlc/dialect/stripe/affine_poly.h"
#include "pmlc/dialect/stripe/types.h"
#include "pmlc/util/enums.h"

namespace pmlc {
namespace dialect {
namespace stripe {

using eltwise::DataType;
using eltwise::ScalarType;
using mlir::ArrayAttr;
using mlir::ArrayRef;
using mlir::Attribute;
using mlir::Builder;
using mlir::IndexType;
using mlir::IntegerAttr;
using mlir::Location;
using mlir::LogicalResult;
using mlir::NamedAttribute;
using mlir::Op;
using mlir::OpAsmParser;
using mlir::OpAsmPrinter;
using mlir::Operation;
using mlir::OperationState;
using mlir::OpFoldResult;
using mlir::OpInterface;
using mlir::OwningRewritePatternList;
using mlir::ParseResult;
using mlir::RankedTensorType;
using mlir::Region;
using mlir::ShapedType;
using mlir::StringAttr;
using mlir::StringRef;
using mlir::Type;
using mlir::TypeAttr;
using mlir::Value;
using util::AggregationKind;

namespace OpTrait = mlir::OpTrait;

#include "pmlc/dialect/stripe/ops_interfaces.h.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/stripe/ops.h.inc"

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
