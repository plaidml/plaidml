// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/ir/ops.h"

#include <vector>

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/tile/ir/util.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::tile {

using llvm::SmallVector;

LogicalResult ContractionOp::materializeOperands(OpBuilder &builder) {
  Operation *op = getOperation();
  if (combo() == CombinationKind::cond) {
    auto operands = op->getOpOperands();
    return success(
        succeeded(tile::materializeOperands(
            builder, op,
            llvm::ArrayRef<OpOperand *>{&operands[0], &operands[3]})) &&
        succeeded(tile::materializeOperands(
            builder, op,
            llvm::ArrayRef<OpOperand *>{&operands[1], &operands[2]})));
  }
  return tile::materializeOperands(builder, getOperation());
}

LogicalResult GatherOp::materializeOperands(OpBuilder &builder) {
  Operation *op = getOperation();
  return tile::materializeOperands(builder, op,
                                   op->getOpOperands().take_front());
}

LogicalResult ReshapeOp::materializeOperands(OpBuilder &builder) {
  return tile::materializeOperands(builder, getOperation());
}

LogicalResult ScatterOp::materializeOperands(OpBuilder &builder) {
  Operation *op = getOperation();
  return tile::materializeOperands(builder, op,
                                   op->getOpOperands().take_front());
}

LogicalResult ShapeOp::materializeOperands(OpBuilder &builder) {
  return tile::materializeOperands(builder, getOperation());
}

LogicalResult PragmaOp::materializeOperands(OpBuilder &builder) {
  return tile::materializeOperands(builder, getOperation());
}

// ---- LayerOp ----

Block *LayerOp::getBody() { return &body().front(); }

void LayerOp::build(OpBuilder &builder, OperationState &result, StringRef op,
                    ArrayRef<Value> operands, ArrayRef<Value> innerResults,
                    DictionaryAttr attrs) {
  for (Value value : innerResults) {
    result.types.push_back(value.getType());
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

void printLayerOp(OpAsmPrinter &p, LayerOp op) {
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
  p.printOptionalAttrDict(op.getAttrs(), elidedAttrs);
}

ParseResult parseLayerOp(OpAsmParser &parser, OperationState &result) {
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

// ---- ReshapeOp ----

OpFoldResult ReshapeOp::fold(ArrayRef<Attribute> operands) {
  // reshape(x, x.shape) -> x
  if (tensor().getType() == getType()) {
    return tensor();
  }
  return {};
}

// ---- ContractionOp ----

unsigned ContractionOp::getNumTensors(CombinationKind combo) {
  switch (combo) {
  case CombinationKind::none:
    return 1;
  case CombinationKind::add:
  case CombinationKind::eq:
  case CombinationKind::mul:
    return 2;
  case CombinationKind::cond:
    return 3;
  default:
    throw std::runtime_error("Invalid combination op");
  }
}

void ContractionOp::build(OpBuilder &builder, OperationState &result,
                          Type resultType, Value init, ArrayRef<Value> tensors,
                          AggregationKind agg, CombinationKind combo,
                          AffineMap sink, ArrayRef<AffineMap> srcs,
                          IntegerSet cons, StringRef name) {
  result.addOperands(init);
  result.addOperands(tensors);
  result.addTypes(resultType);
  result.addAttribute("agg",
                      builder.getI64IntegerAttr(static_cast<int64_t>(agg)));
  result.addAttribute("combo",
                      builder.getI64IntegerAttr(static_cast<int64_t>(combo)));
  result.addAttribute(getSinkAttrName(), AffineMapAttr::get(sink));
  result.addAttribute(getSourcesAttrName(),
                      builder.getAffineMapArrayAttr(srcs));
  if (!cons.isEmptyIntegerSet()) {
    result.addAttribute(getConstraintsAttrName(), IntegerSetAttr::get(cons));
  }
  if (name.size()) {
    result.addAttribute("name", builder.getStringAttr(name));
  }
}

AffineMap ContractionOp::getSourceMap(unsigned i) {
  return srcs().getValue()[i].cast<AffineMapAttr>().getValue();
}

void ContractionOp::setLowerBounds(ArrayRef<int64_t> bounds) {
  SmallVector<AffineExpr, 6> exprs;
  for (auto dim : bounds) {
    exprs.push_back(getAffineConstantExpr(dim, getContext()));
  }
  auto map =
      AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0, exprs, getContext());
  setAttr(getLowerBoundsAttrName(), AffineMapAttr::get(map));
}

void ContractionOp::setUpperBounds(ArrayRef<int64_t> bounds) {
  SmallVector<AffineExpr, 6> exprs;
  for (auto dim : bounds) {
    exprs.push_back(getAffineConstantExpr(dim, getContext()));
  }
  auto map =
      AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0, exprs, getContext());
  setAttr(getUpperBoundsAttrName(), AffineMapAttr::get(map));
}

void ContractionOp::setSink(AffineMap sink) {
  setAttr(getSinkAttrName(), AffineMapAttr::get(sink));
}

void ContractionOp::setSources(ArrayRef<AffineMap> srcs) {
  SmallVector<Attribute, 4> attrs;
  for (auto src : srcs) {
    attrs.push_back(AffineMapAttr::get(src));
  }
  setAttr(getSourcesAttrName(), ArrayAttr::get(attrs, getContext()));
}

void ContractionOp::setConstraints(IntegerSet cons) {
  if (cons.isEmptyIntegerSet()) {
    removeAttr(getConstraintsAttrName());
  } else {
    setAttr(getConstraintsAttrName(), IntegerSetAttr::get(cons));
  }
}

unsigned ContractionOp::getNumTensors() { return getNumTensors(combo()); }

unsigned ContractionOp::getNumSymbols() {
  return getNumOperands() - 1 - getNumTensors();
}

Value ContractionOp::getTensor(unsigned i) {
  return *std::next(operands().begin(), i);
}

Value ContractionOp::getSymbol(unsigned i) {
  return *std::next(operands().begin(), getNumTensors() + i);
}

void printContractionOp(OpAsmPrinter *printer, ContractionOp op) {
  SmallVector<StringRef, 3> elidedAttrs = {"agg", "combo", "name"};
  *printer << op.getOperation()->getName() << ' ';
  *printer << util::stringifyAggregationKind(op.agg());
  *printer << ", ";
  *printer << util::stringifyCombinationKind(op.combo());
  *printer << ", ";
  printer->printOperand(op.init());
  auto numTensors = op.getNumTensors();
  for (unsigned i = 0; i < numTensors; i++) {
    *printer << ", ";
    printer->printOperand(op.getTensor(i));
  }
  auto numSymbols = op.getNumSymbols();
  if (numSymbols) {
    *printer << " [";
    for (unsigned i = 0; i < numSymbols; i++) {
      if (i) {
        *printer << ", ";
      }
      printer->printOperand(op.getSymbol(i));
    }
    *printer << ']';
  }
  printer->printOptionalAttrDict(op.getAttrs(), elidedAttrs);
  *printer << " : ";
  printer->printType(op.init().getType());
  *printer << ", ";
  for (unsigned i = 0; i < numTensors; i++) {
    if (i) {
      *printer << ", ";
    }
    printer->printType(op.getTensor(i).getType());
  }
  *printer << " -> ";
  printer->printType(op.result().getType());
}

ParseResult parseContractionOp(OpAsmParser *parser, OperationState &result) {
  StringRef strAgg;
  StringRef strCombo;
  OpAsmParser::OperandType init;
  SmallVector<OpAsmParser::OperandType, 3> tensors;
  SmallVector<OpAsmParser::OperandType, 8> symbols;
  SmallVector<Type, 4> types;
  Type resultType;
  if (parser->parseKeyword(&strAgg) || parser->parseComma() ||
      parser->parseKeyword(&strCombo) || parser->parseComma() ||
      parser->parseOperand(init) || parser->parseComma()) {
    return failure();
  }

  auto agg = util::symbolizeAggregationKind(strAgg);
  if (!agg) {
    return failure();
  }
  result.addAttribute("agg", parser->getBuilder().getI64IntegerAttr(
                                 static_cast<int64_t>(agg.getValue())));

  auto combo = util::symbolizeCombinationKind(strCombo);
  if (!combo) {
    return failure();
  }
  result.addAttribute("combo", parser->getBuilder().getI64IntegerAttr(
                                   static_cast<int64_t>(combo.getValue())));

  auto numTensors = ContractionOp::getNumTensors(combo.getValue());
  if (parser->parseOperandList(tensors, numTensors) ||
      parser->parseOperandList(symbols,
                               OpAsmParser::Delimiter::OptionalSquare) ||
      parser->parseOptionalAttrDict(result.attributes) ||
      parser->parseColonTypeList(types) || parser->parseArrow() ||
      parser->parseType(resultType)) {
    return failure();
  }

  // TODO: parse a FunctionType here

  auto loc = parser->getCurrentLocation();
  auto indexType = parser->getBuilder().getIndexType();
  auto tensorTypes = llvm::makeArrayRef(types).drop_front();
  if (parser->resolveOperand(init, types.front(), result.operands) ||
      parser->resolveOperands(tensors, tensorTypes, loc, result.operands) ||
      parser->resolveOperands(symbols, indexType, result.operands)) {
    return failure();
  }

  result.addTypes(resultType);
  return success();
}

bool isAnyScalar(Type type) {
  return type.isIndex() || type.isa<FloatType>() || type.isInteger(1) ||
         type.isSignedInteger() || type.isUnsignedInteger();
}

bool isEltwiseAny(Type type) {
  if (auto rankedTensorType = type.dyn_cast<RankedTensorType>()) {
    auto elementType = rankedTensorType.getElementType();
    return isAnyScalar(elementType);
  }
  return isAnyScalar(type);
}

LogicalResult verifyContractionOp(ContractionOp op) {
  auto numTensors = op.getNumTensors();
  auto numSymbols = op.getNumSymbols();
  SmallVector<Value, 8> variadic(op.operands());
  if (variadic.size() < numTensors) {
    return op.emitOpError("combo '")
           << util::stringifyCombinationKind(op.combo()) << "' requires "
           << numTensors << " tensor operands";
  }
  auto shape = op.shape();
  auto resultType = op.result().getType().cast<RankedTensorType>();
  if (!resultType.hasStaticShape() && !shape.hasValue()) {
    return op.emitOpError(
        "attribute 'shape' is required when result type is dynamic");
  }
  unsigned expectedSymbols = op.sink().getNumSymbols();
  if (shape.hasValue()) {
    expectedSymbols += shape->getNumSymbols();
  }
  for (auto src : op.srcs()) {
    auto map = src.cast<AffineMapAttr>();
    expectedSymbols += map.getValue().getNumSymbols();
  }
  if (op.cons().hasValue()) {
    expectedSymbols += op.cons().getValue().getNumSymbols();
  }
  if (expectedSymbols != numSymbols) {
    return op.emitOpError("has incorrect number of symbols: expected ")
           << expectedSymbols << " but found " << numSymbols;
  }
  for (unsigned i = 0; i < numTensors; i++) {
    auto type = op.getTensor(i).getType();
    if (!isEltwiseAny(type)) {
      return op.emitOpError("tensor #")
             << i << " must be eltwise-any, but got " << type;
    }
  }
  for (unsigned i = 0; i < numSymbols; i++) {
    auto type = op.getSymbol(i).getType();
    if (!type.isa<IndexType>()) {
      return op.emitOpError("symbol #")
             << i << " must be index, but got " << type;
    }
  }
  return success();
}

} // namespace pmlc::dialect::tile

#define GET_OP_CLASSES
#include "pmlc/dialect/tile/ir/ops.cc.inc"

#include "pmlc/dialect/tile/ir/interfaces.cc.inc"
