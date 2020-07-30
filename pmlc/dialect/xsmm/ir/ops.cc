// Copyright 2020 Intel Corporation

#include "pmlc/dialect/xsmm/ir/ops.h"

#include <string>

#include "mlir/IR/OpImplementation.h"

namespace pmlc::dialect::xsmm {

using llvm::SmallVector;
using mlir::failure;
using mlir::FunctionType;
using mlir::success;

XSMMDialect::XSMMDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx) {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/xsmm/ir/ops.cc.inc"
      >();
}

//
// ---- GemmInvokeOp ----
//

GemmInvokeOp::operand_range GemmInvokeOp::getOperandsForA() {
  auto aType = a().getType().cast<MemRefType>();
  auto cType = c().getType().cast<MemRefType>();
  return getOperands().slice(4 + cType.getRank(), aType.getRank());
}

GemmInvokeOp::operand_range GemmInvokeOp::getOperandsForB() {
  auto aType = a().getType().cast<MemRefType>();
  auto bType = b().getType().cast<MemRefType>();
  auto cType = c().getType().cast<MemRefType>();
  return getOperands().slice(4 + cType.getRank() + aType.getRank(),
                             bType.getRank());
}

GemmInvokeOp::operand_range GemmInvokeOp::getOperandsForC() {
  auto cType = c().getType().cast<MemRefType>();
  return getOperands().slice(4, cType.getRank());
}

void printGemmInvokeOp(OpAsmPrinter &p, GemmInvokeOp op) {
  auto funcType = FunctionType::get({op.a().getType(), op.b().getType()},
                                    {op.c().getType()}, op.getContext());
  p << op.getOperation()->getName() << ' ';
  p << op.ptr() << ", ";
  p << op.c() << '[';
  p.printOperands(op.getOperandsForC());
  p << "] = " << op.a() << '[';
  p.printOperands(op.getOperandsForA());
  p << "], " << op.b() << '[';
  p.printOperands(op.getOperandsForB());
  p << "] : " << funcType;
}

struct GemmOperand {
  OpAsmParser::OperandType memref;
  SmallVector<OpAsmParser::OperandType, 4> indices;
};

ParseResult parseGemmInvokeOp(OpAsmParser &parser, OperationState &result) {
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

#define GET_OP_CLASSES
#include "pmlc/dialect/xsmm/ir/ops.cc.inc" // NOLINT

} // namespace pmlc::dialect::xsmm
