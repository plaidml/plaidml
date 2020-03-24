// Copyright 2019, Intel Corporation

#pragma once

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffects.h"

#include "pmlc/dialect/eltwise/ir/util.h"
#include "pmlc/util/interfaces.h"

namespace pmlc::dialect::eltwise {

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
using mlir::MemoryEffectOpInterface;
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
using mlir::SmallVectorImpl;
using mlir::StringRef;
using mlir::TensorType;
using mlir::Type;
using mlir::TypeAttr;
using mlir::Value;
using mlir::ValueRange;
using mlir::VectorType;
using util::GenericBuilder;

namespace MemoryEffects = mlir::MemoryEffects;
namespace OpTrait = mlir::OpTrait;
namespace SideEffects = mlir::SideEffects;

#include "pmlc/dialect/eltwise/ir/interfaces.h.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/eltwise/ir/ops.h.inc"

} // namespace pmlc::dialect::eltwise
