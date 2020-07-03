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
  return getOperands().slice(4 + cAccessMap().getNumInputs(),
                             aAccessMap().getNumInputs());
}

GemmInvokeOp::operand_range GemmInvokeOp::getOperandsForB() {
  return getOperands().slice(4 + cAccessMap().getNumInputs() +
                                 aAccessMap().getNumInputs(),
                             bAccessMap().getNumInputs());
}

GemmInvokeOp::operand_range GemmInvokeOp::getOperandsForC() {
  return getOperands().slice(4, cAccessMap().getNumInputs());
}

void printGemmInvokeOp(OpAsmPrinter &p, GemmInvokeOp op) {
  auto funcType = FunctionType::get({op.a().getType(), op.b().getType()},
                                    {op.c().getType()}, op.getContext());
  p << op.getOperation()->getName() << ' ';
  p << op.ptr() << ", ";
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
  p << ", " << op.tile() << " : " << funcType;
}

struct GemmOperandParser {
  OpAsmParser::OperandType operand;
  SmallVector<OpAsmParser::OperandType, 4> accessOperands;
  AffineMapAttr accessMapAttr;
  AffineMapAttr tileMapAttr;
  std::string accessMapAttrName;
  std::string tileMapAttrName;

  explicit GemmOperandParser(StringRef name)
      : accessMapAttrName(name.str() + "AccessMap"),
        tileMapAttrName(name.str() + "TileMap") {}

  ParseResult parse(OpAsmParser &parser, OperationState &result) {
    return failure(
        parser.parseOperand(operand) ||
        parser.parseAffineMapOfSSAIds(accessOperands, accessMapAttr,
                                      accessMapAttrName, result.attributes) ||
        parser.parseColon() ||
        parser.parseAttribute(tileMapAttr, tileMapAttrName, result.attributes));
  }
};

ParseResult parseGemmInvokeOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexType = builder.getIndexType();
  auto i64Type = builder.getIntegerType(64);
  GemmOperandParser a("a"), b("b"), c("c");
  OpAsmParser::OperandType ptr;
  ArrayAttr tileAttr;
  FunctionType funcType;
  return failure(
      parser.parseOperand(ptr) || //
      parser.parseComma() ||      //
      c.parse(parser, result) ||  //
      parser.parseEqual() ||      //
      a.parse(parser, result) ||  //
      parser.parseComma() ||      //
      b.parse(parser, result) ||  //
      parser.parseComma() ||
      parser.parseAttribute(tileAttr, i64Type, "tile", result.attributes) ||
      parser.parseColonType(funcType) ||
      parser.addTypesToList(funcType.getResults(), result.types) ||
      parser.resolveOperand(ptr, i64Type, result.operands) ||
      parser.resolveOperand(c.operand, funcType.getResult(0),
                            result.operands) ||
      parser.resolveOperand(a.operand, funcType.getInput(0), result.operands) ||
      parser.resolveOperand(b.operand, funcType.getInput(1), result.operands) ||
      parser.resolveOperands(c.accessOperands, indexType, result.operands) ||
      parser.resolveOperands(a.accessOperands, indexType, result.operands) ||
      parser.resolveOperands(b.accessOperands, indexType, result.operands));
}

#define GET_OP_CLASSES
#include "pmlc/dialect/xsmm/ir/ops.cc.inc" // NOLINT

} // namespace pmlc::dialect::xsmm
