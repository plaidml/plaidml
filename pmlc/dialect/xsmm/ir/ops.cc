// Copyright 2020 Intel Corporation

#include "pmlc/dialect/xsmm/ir/ops.h"

#include "mlir/IR/OpImplementation.h"

namespace pmlc::dialect::xsmm {

using llvm::SmallVector;
using mlir::failure;
using mlir::success;

//
// ---- GemmOp ----
//

GemmOp::operand_range GemmOp::getOperandsForA() {
  return getOperands().slice(3 + cMap().getNumInputs(), aMap().getNumInputs());
}

GemmOp::operand_range GemmOp::getOperandsForB() {
  return getOperands().slice(3 + cMap().getNumInputs() + aMap().getNumInputs(),
                             bMap().getNumInputs());
}

GemmOp::operand_range GemmOp::getOperandsForC() {
  return getOperands().slice(3, cMap().getNumInputs());
}

void printGemmOp(OpAsmPrinter &p, GemmOp op) {
  p << op.getOperation()->getName() << ' ';
  p << op.c() << '[';
  p.printAffineMapOfSSAIds(op.cMapAttr(), op.getOperandsForC());
  p << "]:" << op.ldc() << " = " << op.a() << '[';
  p.printAffineMapOfSSAIds(op.aMapAttr(), op.getOperandsForA());
  p << "]:" << op.lda() << ", " << op.b() << '[';
  p.printAffineMapOfSSAIds(op.bMapAttr(), op.getOperandsForB());
  p << "]:" << op.ldb() << ", " << op.tile() << " : " << op.c().getType()
    << ", " << op.a().getType() << ", " << op.b().getType();
}

ParseResult parseGemmOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexType = builder.getIndexType();
  auto i64Type = builder.getIntegerType(64);
  SmallVector<Type, 3> operandTypes;
  OpAsmParser::OperandType a, b, c;
  AffineMapAttr aMapAttr, bMapAttr, cMapAttr;
  SmallVector<OpAsmParser::OperandType, 4> aOperands, bOperands, cOperands;
  IntegerAttr ldaAttr, ldbAttr, ldcAttr;
  ArrayAttr tileAttr;
  return failure(
      parser.parseOperand(c) ||
      parser.parseAffineMapOfSSAIds(cOperands, cMapAttr, "cMap",
                                    result.attributes) ||
      parser.parseColon() ||
      parser.parseAttribute(ldcAttr, i64Type, "ldc", result.attributes) ||
      parser.parseEqual() || parser.parseOperand(a) ||
      parser.parseAffineMapOfSSAIds(aOperands, aMapAttr, "aMap",
                                    result.attributes) ||
      parser.parseColon() ||
      parser.parseAttribute(ldaAttr, i64Type, "lda", result.attributes) ||
      parser.parseComma() || parser.parseOperand(b) ||
      parser.parseAffineMapOfSSAIds(bOperands, bMapAttr, "bMap",
                                    result.attributes) ||
      parser.parseColon() ||
      parser.parseAttribute(ldbAttr, i64Type, "ldb", result.attributes) ||
      parser.parseComma() ||
      parser.parseAttribute(tileAttr, i64Type, "tile", result.attributes) ||
      parser.parseColonTypeList(operandTypes) ||
      parser.resolveOperand(c, operandTypes[0], result.operands) ||
      parser.resolveOperand(a, operandTypes[1], result.operands) ||
      parser.resolveOperand(b, operandTypes[2], result.operands) ||
      parser.resolveOperands(cOperands, indexType, result.operands) ||
      parser.resolveOperands(aOperands, indexType, result.operands) ||
      parser.resolveOperands(bOperands, indexType, result.operands));
}

LogicalResult verifyGemmOp(GemmOp op) { return success(); }

#define GET_OP_CLASSES
#include "pmlc/dialect/xsmm/ir/ops.cc.inc"

} // namespace pmlc::dialect::xsmm
