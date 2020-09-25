// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "pmlc/util/enums.h"

namespace pmlc::dialect::pxa {

using llvm::SmallVectorImpl;
using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::AffineMapAttr;
using mlir::AffineReadOpInterface;
using mlir::AffineValueMap;
using mlir::AffineWriteOpInterface;
using mlir::AffineYieldOp;
using mlir::ArrayAttr;
using mlir::ArrayRef;
using mlir::AtomicRMWKind;
using mlir::Attribute;
using mlir::BoolAttr;
using mlir::DictionaryAttr;
using mlir::FloatType;
using mlir::IndexType;
using mlir::IntegerAttr;
using mlir::IntegerType;
using mlir::Location;
using mlir::LogicalResult;
using mlir::LoopLikeOpInterface;
using mlir::MemoryEffectOpInterface;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::NamedAttribute;
using mlir::Op;
using mlir::OpAsmParser;
using mlir::OpAsmPrinter;
using mlir::OpAsmSetValueNameFn;
using mlir::OpBuilder;
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
using mlir::ValueRange;
using mlir::VectorType;

namespace MemoryEffects = mlir::MemoryEffects;
namespace OpTrait = mlir::OpTrait;
namespace SideEffects = mlir::SideEffects;

#define GET_OP_CLASSES
#include "pmlc/dialect/pxa/ir/ops.h.inc"

#include "pmlc/dialect/pxa/ir/dialect.h.inc"

} // namespace pmlc::dialect::pxa
