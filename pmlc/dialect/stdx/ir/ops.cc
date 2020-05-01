// Copyright 2020 Intel Corporation

#include "pmlc/dialect/stdx/ir/ops.h"

#include "mlir/IR/OpImplementation.h"

namespace pmlc::dialect::stdx {

using mlir::failure;
using mlir::success;

namespace {

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

// ---- ReshapeOp ----

void printReshapeOp(OpAsmPrinter &p, ReshapeOp op) {
  p << op.getOperation()->getName() << " (" << op.getOperand(0) << ", "
    << op.getOperand(1) << ") : (" << op.getOperand(0).getType() << ", "
    << op.getOperand(1).getType() << ") -> "
    << op.getOperation()->getResult(0).getType();
}

ParseResult parseReshapeOp(OpAsmParser &parser, OperationState &result) {
  return failure();
}

LogicalResult verifyReshapeOp(ReshapeOp op) { return success(); }

static void buildReshapeOp(Builder *builder, OperationState &result,
                           Value tensor, ArrayRef<Value> dims,
                           MemRefType destType) {
  result.addOperands(tensor);
  result.addOperands(dims);
  result.addTypes(destType);
}

} // namespace

#define GET_OP_CLASSES
#include "pmlc/dialect/stdx/ir/ops.cc.inc"

StdXDialect::StdXDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx) {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/stdx/ir/ops.cc.inc" // NOLINT
      >();
}

} // namespace pmlc::dialect::stdx
