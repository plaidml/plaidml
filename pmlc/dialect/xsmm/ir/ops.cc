// Copyright 2020 Intel Corporation

#include "pmlc/dialect/xsmm/ir/ops.h"

#include <string>

#include "mlir/IR/OpImplementation.h"

#include "pmlc/dialect/xsmm/ir/enums.cc.inc"

using namespace mlir; // NOLINT

namespace pmlc::dialect::xsmm {

void XSMMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/xsmm/ir/ops.cc.inc"
      >();
}

struct GemmOperand {
  OpAsmParser::OperandType memref;
  SmallVector<OpAsmParser::OperandType, 4> indices;
};

//
// ---- GemmInvokeF32Op ----
//

GemmInvokeF32Op::operand_range GemmInvokeF32Op::getOperandsForA() {
  auto aType = a().getType().cast<MemRefType>();
  auto cType = c().getType().cast<MemRefType>();
  return getOperands().slice(4 + cType.getRank(), aType.getRank());
}

GemmInvokeF32Op::operand_range GemmInvokeF32Op::getOperandsForB() {
  auto aType = a().getType().cast<MemRefType>();
  auto bType = b().getType().cast<MemRefType>();
  auto cType = c().getType().cast<MemRefType>();
  return getOperands().slice(4 + cType.getRank() + aType.getRank(),
                             bType.getRank());
}

GemmInvokeF32Op::operand_range GemmInvokeF32Op::getOperandsForC() {
  auto cType = c().getType().cast<MemRefType>();
  return getOperands().slice(4, cType.getRank());
}

void printGemmInvokeF32Op(OpAsmPrinter &p, GemmInvokeF32Op op) {
  auto funcType =
      FunctionType::get(op.getContext(), {op.a().getType(), op.b().getType()},
                        {op.c().getType()});
  p << ' ' << op.ptr() << ", ";
  p << op.c() << '[';
  p.printOperands(op.getOperandsForC());
  p << "] = " << op.a() << '[';
  p.printOperands(op.getOperandsForA());
  p << "], " << op.b() << '[';
  p.printOperands(op.getOperandsForB());
  p << "] : " << funcType;
}

ParseResult parseGemmInvokeF32Op(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexType = builder.getIndexType();
  auto i64Type = builder.getIntegerType(64);
  GemmOperand a, b, c;
  OpAsmParser::OperandType ptr;
  FunctionType funcType;
  return failure(
      parser.parseOperand(ptr) || parser.parseComma() ||
      parser.parseOperand(c.memref) ||
      parser.parseOperandList(c.indices, OpAsmParser::Delimiter::Square) ||
      parser.parseEqual() || parser.parseOperand(a.memref) ||
      parser.parseOperandList(a.indices, OpAsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseOperand(b.memref) ||
      parser.parseOperandList(b.indices, OpAsmParser::Delimiter::Square) ||
      parser.parseColonType(funcType) ||
      parser.resolveOperand(ptr, i64Type, result.operands) ||
      parser.resolveOperand(c.memref, funcType.getResult(0), result.operands) ||
      parser.resolveOperand(a.memref, funcType.getInput(0), result.operands) ||
      parser.resolveOperand(b.memref, funcType.getInput(1), result.operands) ||
      parser.resolveOperands(c.indices, indexType, result.operands) ||
      parser.resolveOperands(a.indices, indexType, result.operands) ||
      parser.resolveOperands(b.indices, indexType, result.operands));
}

//
// ---- BRGemmOffsInvokeF32Op ----
//

ParseResult parseBRGemmOffsInvokeF32Op(OpAsmParser &parser,
                                       OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexType = builder.getIndexType();
  auto i64Type = builder.getIntegerType(64);
  GemmOperand a, b, c;
  ArrayAttr aOffs, bOffs;
  IntegerAttr numBatchesAttr;
  FunctionType funcType;
  OpAsmParser::OperandType ptr;
  return failure(
      parser.parseOperand(ptr) || parser.parseComma() ||
      parser.parseOperand(c.memref) ||
      parser.parseOperandList(c.indices, OpAsmParser::Delimiter::Square) ||
      parser.parseEqual() || parser.parseOperand(a.memref) ||
      parser.parseOperandList(a.indices, OpAsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseOperand(b.memref) ||
      parser.parseOperandList(b.indices, OpAsmParser::Delimiter::Square) ||
      parser.parseComma() ||
      parser.parseAttribute(numBatchesAttr, i64Type, "numBatches",
                            result.attributes) ||
      parser.parseComma() ||
      parser.parseAttribute(aOffs, i64Type, "aOffsets", result.attributes) ||
      parser.parseComma() ||
      parser.parseAttribute(bOffs, i64Type, "bOffsets", result.attributes) ||
      parser.parseColonType(funcType) ||
      parser.resolveOperand(ptr, i64Type, result.operands) ||
      parser.resolveOperand(c.memref, funcType.getResult(0), result.operands) ||
      parser.resolveOperand(a.memref, funcType.getInput(0), result.operands) ||
      parser.resolveOperand(b.memref, funcType.getInput(1), result.operands) ||
      parser.resolveOperands(c.indices, indexType, result.operands) ||
      parser.resolveOperands(a.indices, indexType, result.operands) ||
      parser.resolveOperands(b.indices, indexType, result.operands));
}

//
// ---- UnaryInvokeOp ----
//

struct UnaryOperand {
  OpAsmParser::OperandType memref;
  SmallVector<OpAsmParser::OperandType, 4> indices;
};

Operation::operand_range UnaryInvokeOp::getOperandsForOutput() {
  auto outputType = output().getType().cast<MemRefType>();
  return getOperands().slice(3, outputType.getRank());
}

Operation::operand_range UnaryInvokeOp::getOperandsForInput() {
  auto inputType = input().getType().dyn_cast<MemRefType>();
  auto outputType = output().getType().cast<MemRefType>();

  if (inputType)
    return getOperands().slice(3 + outputType.getRank(), inputType.getRank());
  else // scalar
    return getOperands().slice(3 + outputType.getRank(), 0);
}

void printUnaryInvokeOp(OpAsmPrinter &p, UnaryInvokeOp op) {
  auto funcType = FunctionType::get(op.getContext(), {op.input().getType()},
                                    {op.output().getType()});
  p << ' ' << op.output() << '[';
  p.printOperands(op.getOperandsForOutput());
  p << "] = " << op.ptr() << '(' << op.input() << '[';
  p.printOperands(op.getOperandsForInput());
  p << "]) : " << funcType;
}

ParseResult parseUnaryInvokeOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexType = builder.getIndexType();
  auto i64Type = builder.getIntegerType(64);
  UnaryOperand input, output;
  OpAsmParser::OperandType ptr;
  FunctionType funcType;
  return failure(
      parser.parseOperand(output.memref) ||
      parser.parseOperandList(output.indices, OpAsmParser::Delimiter::Square) ||
      parser.parseEqual() || parser.parseOperand(ptr) || parser.parseLParen() ||
      parser.parseOperand(input.memref) ||
      parser.parseOperandList(input.indices, OpAsmParser::Delimiter::Square) ||
      parser.parseRParen() || parser.parseColonType(funcType) ||
      parser.resolveOperand(ptr, i64Type, result.operands) ||
      parser.resolveOperand(output.memref, funcType.getResult(0),
                            result.operands) ||
      parser.resolveOperand(input.memref, funcType.getInput(0),
                            result.operands) ||
      parser.resolveOperands(output.indices, indexType, result.operands) ||
      parser.resolveOperands(input.indices, indexType, result.operands));
}

} // namespace pmlc::dialect::xsmm

#include "pmlc/dialect/xsmm/ir/dialect.cc.inc" // NOLINT

#define GET_OP_CLASSES
#include "pmlc/dialect/xsmm/ir/ops.cc.inc" // NOLINT
