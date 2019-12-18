// Copyright 2019, Intel Corporation

#pragma once

#include "pmlc/util/enums.h"

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Transforms/LoopLikeInterface.h"

namespace pmlc {
namespace dialect {
namespace pxa {

using mlir::AffineMap;
using mlir::AffineMapAttr;
using mlir::AffineTerminatorOp;
using mlir::ArrayAttr;
using mlir::ArrayRef;
using mlir::Attribute;
using mlir::BoolAttr;
using mlir::Builder;
using mlir::FloatType;
using mlir::IndexType;
using mlir::IntegerAttr;
using mlir::IntegerType;
using mlir::Location;
using mlir::LogicalResult;
using mlir::LoopLikeOpInterface;
using mlir::MemRefType;
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

#define GET_OP_CLASSES
#include "pmlc/dialect/pxa/ops.h.inc"

}  // namespace pxa
}  // namespace dialect
}  // namespace pmlc
