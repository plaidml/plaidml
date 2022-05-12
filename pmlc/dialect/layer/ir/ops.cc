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
                  ValueRange operands, TypeRange resultTypes,
                  DictionaryAttr attrs) {
  result.addTypes(resultTypes);
  result.addOperands(operands);
  result.addAttribute("op", builder.getStringAttr(op));
  result.addAttribute("attrs", attrs);
  // Create a region and a block for the body.
  Region *bodyRegion = result.addRegion();
  auto *body = new Block();
  // Add all the block arguments.
  for (Value operand : operands) {
    body->addArgument(operand.getType(), operand.getLoc());
  }
  bodyRegion->push_back(body);
}
#if 0
void BoxOp::print(OpAsmPrinter &p) {
  p << " \"" << op() << "\" (" << getBody()->getArguments() << ") = (";
  p.printOperands(operands());
  p << ") : ";
  p.printFunctionalType(*this);
  p.printRegion(body(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  SmallVector<StringRef, 2> elidedAttrs{"op"};
  auto att = attrs().cast<DictionaryAttr>();
  if (att.empty()) {
    elidedAttrs.push_back("attrs");
  }
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

ParseResult BoxOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr opName;
  FunctionType funcType;
  SmallVector<OpAsmParser::Argument, 4> inner;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> outer;
  auto loc = parser.getCurrentLocation();
  if (parser.parseAttribute(opName, "op", result.attributes) ||
      parser.parseArgumentList(inner, OpAsmParser::Delimiter::Paren) ||
      parser.parseEqual() ||
      parser.parseOperandList(outer, OpAsmParser::Delimiter::Paren) ||
      parser.parseColonType(funcType) ||
      parser.resolveOperands(outer, funcType.getInputs(), loc,
                             result.operands)) {
    return failure();
  }
  result.addTypes(funcType.getResults());
  Region *body = result.addRegion();
  SmallVector<Type> argTypes;
  if (parser.parseRegion(*body, inner) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  if (!result.attributes.getNamed("attrs")) {
    auto builder = parser.getBuilder();
    result.attributes.set("attrs", builder.getDictionaryAttr({}));
  }
  return success();
}
#endif
void LayerDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/layer/ir/ops.cc.inc"
      >();
}

} // namespace pmlc::dialect::layer

#include "pmlc/dialect/layer/ir/dialect.cc.inc" // NOLINT

#define GET_OP_CLASSES
#include "pmlc/dialect/layer/ir/ops.cc.inc" // NOLINT
