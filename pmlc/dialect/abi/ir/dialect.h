// Copyright 2019, Intel Corporation

#pragma once

/*
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "pmlc/util/enums.h"
 */

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "pmlc/util/loop_with_epilog.h"

namespace pmlc::dialect::abi {

/*
using llvm::SmallVectorImpl;
using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::AffineMapAttr;
using mlir::AffineValueMap;
using mlir::AffineYieldOp;
using mlir::ArrayAttr;
using mlir::AtomicRMWKind;
using mlir::Attribute;
using mlir::BoolAttr;
using mlir::DictionaryAttr;
using mlir::FloatType;
using mlir::IndexType;
using mlir::IntegerAttr;
using mlir::IntegerType;
using mlir::Location;
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
using mlir::OperationState;
using mlir::OpFoldResult;
using mlir::OpInterface;
using mlir::OwningRewritePatternList;
using mlir::ParseResult;
using mlir::RankedTensorType;
using mlir::ShapedType;
using mlir::StringAttr;
using mlir::StringRef;
using mlir::Type;
using mlir::TypeAttr;
using mlir::ValueRange;
using mlir::VectorType;

namespace MemoryEffects = mlir::MemoryEffects;
namespace SideEffects = mlir::SideEffects;
*/
using mlir::ArrayRef;
using mlir::LogicalResult;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::Region;
using mlir::Value;
using mlir::LLVM::LLVMFuncOp;

namespace OpTrait = mlir::OpTrait;

} // namespace pmlc::dialect::abi

#define GET_OP_CLASSES
#include "pmlc/dialect/abi/ir/ops.h.inc"

#include "pmlc/dialect/abi/ir/dialect.h.inc"
