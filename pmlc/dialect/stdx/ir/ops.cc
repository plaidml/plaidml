// Copyright 2020 Intel Corporation

#include "pmlc/dialect/stdx/ir/ops.h"

#include "mlir/IR/OpImplementation.h"

namespace pmlc::dialect::stdx {

using llvm::SmallVector;
using mlir::failure;
using mlir::success;

namespace {

// ---- AtomicRMWOp ----

void printAtomicRMWOp(OpAsmPrinter &p, AtomicRMWOp op) {
  p << op.getOperation()->getName() << ' ' << op.getInductionVar();
  p << " = " << op.memref() << '[' << op.indices() << ']';
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.memref().getType();
  p.printRegion(op.body(), /*printEntryBlockArgs=*/false);
}

// <operation> ::= `stdx.atomic_rmw` argument `=` ssa-use `[` ssa-use-list `]`
//                 attribute-dict? `:` type region
ParseResult parseAtomicRMWOp(OpAsmParser &parser, OperationState &result) {
  MemRefType type;
  OpAsmParser::OperandType iv;
  OpAsmParser::OperandType memref;
  auto indexTy = parser.getBuilder().getIndexType();
  SmallVector<OpAsmParser::OperandType, 4> idxs;
  Region *body = result.addRegion();
  if (parser.parseRegionArgument(iv) || parser.parseEqual() ||
      parser.parseOperand(memref) ||
      parser.parseOperandList(idxs, OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(memref, type, result.operands) ||
      parser.resolveOperands(idxs, indexTy, result.operands) ||
      parser.parseRegion(*body, iv, type.getElementType())) {
    return failure();
  }
  result.addTypes(type.getElementType());
  return success();
}

LogicalResult verifyAtomicRMWOp(AtomicRMWOp op) {
  if (op.getMemRefType().getRank() != op.getNumOperands() - 1)
    return op.emitOpError(
        "expects the number of subscripts to be equal to memref rank");
  auto block = op.getBody();
  if (block->empty()) {
    return op.emitOpError("expects a non-empty body");
  }
  auto elementType = op.getMemRefType().getElementType();
  if (block->getNumArguments() != 1 ||
      block->getArgument(0).getType() != elementType) {
    return op.emitOpError()
           << "expects a body with one argument of type " << elementType;
  }
  if (!llvm::isa<AtomicRMWYieldOp>(block->getTerminator())) {
    return op.emitOpError(
        "expects the body to be terminated with a 'stdx.atomic_rmw.yield' op");
  }
  return success();
}

// ---- AtomicRMWYieldOp ----

void printAtomicRMWYieldOp(OpAsmPrinter &p, AtomicRMWYieldOp op) {
  p << op.getOperation()->getName() << ' ';
  p << op.result() << " : ";
  p.printType(op.result().getType());
}

// <operation> ::= `stdx.atomic_rmw.yield` ssa-use `:` type
ParseResult parseAtomicRMWYieldOp(OpAsmParser &parser, OperationState &result) {
  Type type;
  OpAsmParser::OperandType res;
  return failure(parser.parseOperand(res) || parser.parseColonType(type) ||
                 parser.resolveOperand(res, type, result.operands));
}

LogicalResult verifyAtomicRMWYieldOp(AtomicRMWYieldOp op) {
  auto parentOp = op.getParentOfType<AtomicRMWOp>();
  Type elementType = parentOp.getMemRefType().getElementType();
  if (elementType != op.result().getType()) {
    return op.emitOpError() << "needs to have type " << elementType;
  }
  return success();
}

// ---- FPToSIOp ----

void printFPToSIOp(OpAsmPrinter &p, FPToSIOp op) {
  p << op.getOperation()->getName() << ' ' << op.getOperand() << " : "
    << op.getOperand().getType() << " to "
    << op.getOperation()->getResult(0).getType();
}

ParseResult parseFPToSIOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType srcInfo;
  Type srcType, dstType;
  return failure(parser.parseOperand(srcInfo) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 parser.parseColonType(srcType) ||
                 parser.resolveOperand(srcInfo, srcType, result.operands) ||
                 parser.parseKeywordType("to", dstType) ||
                 parser.addTypeToList(dstType, result.types));
}

LogicalResult verifyFPToSIOp(FPToSIOp op) {
  auto opType = op.getOperand().getType();
  auto resType = op.getOperation()->getResult(0).getType();
  if (auto fromFloatType = opType.dyn_cast<mlir::FloatType>()) {
    if (auto intoIntType = resType.dyn_cast<mlir::IntegerType>()) {
      return success();
    }
  }
  return op.emitError("operand type ") << opType << " and result type "
                                       << resType << " are cast incompatible";
}

static void buildFPToSIOp(Builder *builder, OperationState &result,
                          Value source, Type destType) {
  result.addOperands(source);
  result.addTypes(destType);
}

// ---- FPToUIOp ----

void printFPToUIOp(OpAsmPrinter &p, FPToUIOp op) {
  p << op.getOperation()->getName() << ' ' << op.getOperand() << " : "
    << op.getOperand().getType() << " to "
    << op.getOperation()->getResult(0).getType();
}

ParseResult parseFPToUIOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType srcInfo;
  Type srcType, dstType;
  return failure(parser.parseOperand(srcInfo) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 parser.parseColonType(srcType) ||
                 parser.resolveOperand(srcInfo, srcType, result.operands) ||
                 parser.parseKeywordType("to", dstType) ||
                 parser.addTypeToList(dstType, result.types));
}

LogicalResult verifyFPToUIOp(FPToUIOp op) {
  auto opType = op.getOperand().getType();
  auto resType = op.getOperation()->getResult(0).getType();
  if (auto fromFloatType = opType.dyn_cast<mlir::FloatType>()) {
    if (auto intoIntType = resType.dyn_cast<mlir::IntegerType>()) {
      return success();
    }
  }
  return op.emitError("operand type ") << opType << " and result type "
                                       << resType << " are cast incompatible";
}

static void buildFPToUIOp(Builder *builder, OperationState &result,
                          Value source, Type destType) {
  result.addOperands(source);
  result.addTypes(destType);
}

} // namespace

#define GET_OP_CLASSES
#include "pmlc/dialect/stdx/ir/ops.cc.inc"

} // namespace pmlc::dialect::stdx
