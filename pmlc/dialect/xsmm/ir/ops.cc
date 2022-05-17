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
  OpAsmParser::UnresolvedOperand memref;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> indices;
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

void GemmInvokeF32Op::print(OpAsmPrinter &p) {
  auto funcType = FunctionType::get(
      getContext(), {a().getType(), b().getType()}, {c().getType()});
  p << ' ' << ptr() << ", ";
  p << c() << '[';
  p.printOperands(getOperandsForC());
  p << "] = " << a() << '[';
  p.printOperands(getOperandsForA());
  p << "], " << b() << '[';
  p.printOperands(getOperandsForB());
  p << "] : " << funcType;
}

ParseResult GemmInvokeF32Op::parse(OpAsmParser &parser,
                                   OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexType = builder.getIndexType();
  auto i64Type = builder.getIntegerType(64);
  GemmOperand a, b, c;
  OpAsmParser::UnresolvedOperand ptr;
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

ParseResult BRGemmOffsInvokeF32Op::parse(OpAsmParser &parser,
                                         OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexType = builder.getIndexType();
  auto i64Type = builder.getIntegerType(64);
  GemmOperand a, b, c;
  ArrayAttr aOffs, bOffs;
  IntegerAttr numBatchesAttr;
  FunctionType funcType;
  OpAsmParser::UnresolvedOperand ptr;
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

BRGemmOffsInvokeF32Op::operand_range BRGemmOffsInvokeF32Op::getOperandsForA() {
  auto aType = a().getType().cast<MemRefType>();
  auto cType = c().getType().cast<MemRefType>();
  return getOperands().slice(4 + cType.getRank(), aType.getRank());
}

BRGemmOffsInvokeF32Op::operand_range BRGemmOffsInvokeF32Op::getOperandsForB() {
  auto aType = a().getType().cast<MemRefType>();
  auto bType = b().getType().cast<MemRefType>();
  auto cType = c().getType().cast<MemRefType>();
  return getOperands().slice(4 + cType.getRank() + aType.getRank(),
                             bType.getRank());
}

BRGemmOffsInvokeF32Op::operand_range BRGemmOffsInvokeF32Op::getOperandsForC() {
  auto cType = c().getType().cast<MemRefType>();
  return getOperands().slice(4, cType.getRank());
}

void BRGemmOffsInvokeF32Op::print(OpAsmPrinter &p) {
  auto funcType = FunctionType::get(
      getContext(), {a().getType(), b().getType()}, {c().getType()});
  p << ' ' << ptr() << ", ";
  p << c() << '[';
  p.printOperands(getOperandsForC());
  p << "] = " << a() << '[';
  p.printOperands(getOperandsForA());
  p << "], " << b() << '[';
  p.printOperands(getOperandsForB());
  p << "] : " << funcType;
  p << " aOffsets = " << aOffsets();
  p << " bOffsets = " << bOffsets();
  p << " numBatches = " << numBatches();
}

//
// ---- UnaryInvokeOp ----
//

struct UnaryOperand {
  OpAsmParser::UnresolvedOperand memref;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> indices;
};

Operation::operand_range UnaryInvokeOp::getOperandsForOutput() {
  auto outputType = output().getType().cast<MemRefType>();
  return getOperands().slice(3, outputType.getRank());
}

Operation::operand_range UnaryInvokeOp::getOperandsForInput() {
  auto inputType = input().getType().cast<MemRefType>();
  auto outputType = output().getType().cast<MemRefType>();
  return getOperands().slice(3 + outputType.getRank(), inputType.getRank());
}

void UnaryInvokeOp::print(OpAsmPrinter &p) {
  auto funcType = FunctionType::get(getContext(), {input().getType()},
                                    {output().getType()});
  p << ' ' << output() << '[';
  p.printOperands(getOperandsForOutput());
  p << "] = " << ptr() << '(' << input() << '[';
  p.printOperands(getOperandsForInput());
  p << "]) : " << funcType;
}

ParseResult UnaryInvokeOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexType = builder.getIndexType();
  auto i64Type = builder.getIntegerType(64);
  UnaryOperand input, output;
  OpAsmParser::UnresolvedOperand ptr;
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

//
// ---- UnaryInvokeOp ----
//

void BinaryInvokeOp::print(OpAsmPrinter &p) {
  auto funcType =
      FunctionType::get(getContext(), {input1().getType(), input2().getType()},
                        {output().getType()});
  p << ' ' << output() << " = ";
  p << ptr() << '(';
  p.printOperands(ValueRange{input1(), input2()});
  p << ") : " << funcType;
}

ParseResult BinaryInvokeOp::parse(OpAsmParser &parser, OperationState &result) {
  assert(0 && "implement me");
  return failure();
}

//
// --- BinaryDispatchOp ---
//

static std::string stringfy(BinaryKind kind) {
  switch (kind) {
  case BinaryKind::NONE:
    return "none";
  case BinaryKind::ADD:
    return "add";
  case BinaryKind::MUL:
    return "mul";
  case BinaryKind::SUB:
    return "sub";
  case BinaryKind::DIV:
    return "div";
  case BinaryKind::MULADD:
    return "muladd";
  case BinaryKind::MATMUL:
    return "matmul";
  case BinaryKind::MUL_AND_REDUCE_TO_SCALAR_OP_ADD:
    return "mul_and_reduce_to_scalar_op_add";
  case BinaryKind::PACK:
    return "pack";
  }
  llvm_unreachable("kind not right");
}

void BinaryDispatchOp::print(OpAsmPrinter &p) {
  p << '{';
  p << ' ' << "bcastType1 = " << bcastType1();
  p << ' ' << "bcastType2 = " << bcastType2();
  p << ' ' << "compute_type = " << compute_type();
  p << ' ' << "func_type = " << func_type();
  p << ' ' << "kind = " << stringfy(kind());
  p << ' ' << "ldi1 = " << ldi1();
  p << ' ' << "ldi2 = " << ldi2();
  p << ' ' << "ldo = " << ldo();
  p << ' ' << "tile = " << tile();
  p << '}';
}

ParseResult BinaryDispatchOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  assert(0 && "implement me");
  return failure();
}

//
// --- BRGemmInvokeF32Op ---
//

void BRGemmInvokeF32Op::print(OpAsmPrinter &p) {
  // assert(0 && "implement me");
}

ParseResult BRGemmInvokeF32Op::parse(OpAsmParser &parser,
                                     OperationState &result) {
  assert(0 && "implement me");
  return failure();
}

} // namespace pmlc::dialect::xsmm

#include "pmlc/dialect/xsmm/ir/dialect.cc.inc" // NOLINT

#define GET_OP_CLASSES
#include "pmlc/dialect/xsmm/ir/ops.cc.inc" // NOLINT
