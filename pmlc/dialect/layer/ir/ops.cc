// Copyright 2020, Intel Corporation

#include "pmlc/dialect/layer/ir/ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::layer {

using llvm::SmallVector;

Block *BoxOp::getBody() { return &body().front(); }

void BoxOp::build(OpBuilder &builder, OperationState &result, StringRef op,
                  ArrayRef<Value> operands, ArrayRef<Type> resultTypes,
                  DictionaryAttr attrs) {
  for (Type type : resultTypes) {
    result.types.push_back(type);
  }
  result.addOperands(operands);
  result.addAttribute("op", builder.getStringAttr(op));
  result.addAttribute("attrs", attrs);
  // Create a region and a block for the body.
  Region *bodyRegion = result.addRegion();
  auto *body = new Block();
  // Add all the block arguments.
  for (Value operand : operands) {
    body->addArgument(operand.getType());
  }
  bodyRegion->push_back(body);
}

void printBoxOp(OpAsmPrinter &p, BoxOp op) {
  p << op.getOperationName() << " \"" << op.op() << "\" ("
    << op.getBody()->getArguments() << ") = (";
  p.printOperands(op.operands());
  p << ") : ";
  p.printFunctionalType(op);
  p.printRegion(op.body(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  SmallVector<StringRef, 2> elidedAttrs{"op"};
  auto attrs = op.attrs().cast<DictionaryAttr>();
  if (attrs.empty()) {
    elidedAttrs.push_back("attrs");
  }
  p.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
}

ParseResult parseBoxOp(OpAsmParser &parser, OperationState &result) {
  StringAttr opName;
  FunctionType funcType;
  SmallVector<OpAsmParser::OperandType, 4> inner;
  SmallVector<OpAsmParser::OperandType, 4> outer;
  auto loc = parser.getCurrentLocation();
  if (parser.parseAttribute(opName, "op", result.attributes) ||
      parser.parseRegionArgumentList(inner, OpAsmParser::Delimiter::Paren) ||
      parser.parseEqual() || //
      parser.parseOperandList(outer, OpAsmParser::Delimiter::Paren) ||
      parser.parseColonType(funcType) ||
      parser.resolveOperands(outer, funcType.getInputs(), loc,
                             result.operands)) {
    return failure();
  }
  result.addTypes(funcType.getResults());
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, inner, funcType.getInputs()) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  if (!result.attributes.getNamed("attrs")) {
    auto builder = parser.getBuilder();
    result.attributes.set("attrs", builder.getDictionaryAttr({}));
  }
  return success();
}

void LayerDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/layer/ir/ops.cc.inc"
      >();
}

} // namespace pmlc::dialect::layer

#define GET_OP_CLASSES
#include "pmlc/dialect/layer/ir/ops.cc.inc" // NOLINT
