// Copyright 2020 Intel Corporation

#include "pmlc/dialect/stdx/ir/ops.h"

#include "mlir/IR/OpImplementation.h"

namespace pmlc::dialect::stdx {

using mlir::failure;
using mlir::FunctionType;
using mlir::success;

namespace {

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

static void buildFPToUIOp(OpBuilder &builder, OperationState &result,
                          Value source, Type destType) {
  result.addOperands(source);
  result.addTypes(destType);
}

// ---- UIToFPOp ----

void printUIToFPOp(OpAsmPrinter &p, UIToFPOp op) {
  p << op.getOperation()->getName() << ' ' << op.getOperand() << " : "
    << op.getOperand().getType() << " to "
    << op.getOperation()->getResult(0).getType();
}

ParseResult parseUIToFPOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType srcInfo;
  Type srcType, dstType;
  return failure(parser.parseOperand(srcInfo) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 parser.parseColonType(srcType) ||
                 parser.resolveOperand(srcInfo, srcType, result.operands) ||
                 parser.parseKeywordType("to", dstType) ||
                 parser.addTypeToList(dstType, result.types));
}

LogicalResult verifyUIToFPOp(UIToFPOp op) {
  auto opType = op.getOperand().getType();
  auto resType = op.getOperation()->getResult(0).getType();
  if (auto fromIntType = opType.dyn_cast<mlir::IntegerType>()) {
    if (auto intoFloatType = resType.dyn_cast<mlir::FloatType>()) {
      return success();
    }
  }
  return op.emitError("operand type ") << opType << " and result type "
                                       << resType << " are cast incompatible";
}

static void buildUIToFPOp(OpBuilder &builder, OperationState &result,
                          Value source, Type destType) {
  result.addOperands(source);
  result.addTypes(destType);
}

} // namespace

// ---- ReshapeOp ----

Value ReshapeOp::getViewSource() { return tensor(); }

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
