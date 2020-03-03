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
  return getOperands().slice(3 + cAccessMap().getNumInputs(),
                             aAccessMap().getNumInputs());
}

GemmOp::operand_range GemmOp::getOperandsForB() {
  return getOperands().slice(3 + cAccessMap().getNumInputs() +
                                 aAccessMap().getNumInputs(),
                             bAccessMap().getNumInputs());
}

GemmOp::operand_range GemmOp::getOperandsForC() {
  return getOperands().slice(3, cAccessMap().getNumInputs());
}

void printGemmOp(OpAsmPrinter &p, GemmOp op) {
  p << op.getOperation()->getName() << ' ';
  p << op.c() << '[';
  p.printAffineMapOfSSAIds(op.cAccessMapAttr(), op.getOperandsForC());
  p << "]:";
  p.printAttribute(op.cTileMapAttr());
  p << " = " << op.a() << '[';
  p.printAffineMapOfSSAIds(op.aAccessMapAttr(), op.getOperandsForA());
  p << "]:";
  p.printAttribute(op.aTileMapAttr());
  p << ", " << op.b() << '[';
  p.printAffineMapOfSSAIds(op.bAccessMapAttr(), op.getOperandsForB());
  p << "]:";
  p.printAttribute(op.bTileMapAttr());
  p << ", " << op.tile() << " : " << op.c().getType() << ", "
    << op.a().getType() << ", " << op.b().getType();
}

ParseResult parseGemmOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexType = builder.getIndexType();
  auto i64Type = builder.getIntegerType(64);
  SmallVector<Type, 3> operandTypes;
  OpAsmParser::OperandType a, b, c;
  AffineMapAttr aAccessMapAttr, bAccessMapAttr, cAccessMapAttr;
  AffineMapAttr aTileMapAttr, bTileMapAttr, cTileMapAttr;
  SmallVector<OpAsmParser::OperandType, 4> aOperands, bOperands, cOperands;
  ArrayAttr tileAttr;
  return failure(
      parser.parseOperand(c) ||
      parser.parseAffineMapOfSSAIds(cOperands, cAccessMapAttr, "cAccessMap",
                                    result.attributes) ||
      parser.parseColon() ||
      parser.parseAttribute(cTileMapAttr, "cTileMap", result.attributes) ||
      parser.parseEqual() || parser.parseOperand(a) ||
      parser.parseAffineMapOfSSAIds(aOperands, aAccessMapAttr, "aAccessMap",
                                    result.attributes) ||
      parser.parseColon() ||
      parser.parseAttribute(aTileMapAttr, "aTileMap", result.attributes) ||
      parser.parseComma() || parser.parseOperand(b) ||
      parser.parseAffineMapOfSSAIds(bOperands, bAccessMapAttr, "bAccessMap",
                                    result.attributes) ||
      parser.parseColon() ||
      parser.parseAttribute(bTileMapAttr, "bTileMap", result.attributes) ||
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
