// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/ir/ops.h"

#include <vector>

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/eltwise/ir/util.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

namespace pmlc::dialect::tile {

using llvm::SmallVector;
using mlir::AffineExpr;
using mlir::ArrayAttr;
using mlir::failure;
using mlir::FloatAttr;
using mlir::IntegerAttr;
using mlir::IntegerType;
using mlir::LogicalResult;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::StringAttr;
using mlir::success;
using mlir::Value;

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  // IVLOG(5, "ConstantOp::fold> " << mlir::debugString(*getOperation()));
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

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
                          IntegerSet cons, bool no_reduce, StringRef name) {
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
  if (no_reduce) {
    result.addAttribute("no_reduce", builder.getUnitAttr());
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
    exprs.push_back(mlir::getAffineConstantExpr(dim, getContext()));
  }
  auto map =
      AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0, exprs, getContext());
  setAttr(getLowerBoundsAttrName(), AffineMapAttr::get(map));
}

void ContractionOp::setUpperBounds(ArrayRef<int64_t> bounds) {
  SmallVector<AffineExpr, 6> exprs;
  for (auto dim : bounds) {
    exprs.push_back(mlir::getAffineConstantExpr(dim, getContext()));
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

// --- GatherOp ---

struct GatherCanonicalizer : public OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp gatherOp,
                                PatternRewriter &rewriter) const override {
    IVLOG(5, "IndexCanonicalizer::matchAndRewrite> "
                 << mlir::debugString(gatherOp));
    auto op = gatherOp.getOperation();
    SmallVector<Value, 2> operands(op->getOperands());
    auto resultType = GatherOp::getResultType(operands);
    if (resultType == gatherOp.result().getType()) {
      return failure();
    }
    auto newOp = rewriter.create<GatherOp>(op->getLoc(), resultType,
                                           gatherOp.tensor(), gatherOp.dims());
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
    return success();
  }
};

void GatherOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<GatherCanonicalizer>(context);
}

Type GatherOp::getResultType(ArrayRef<Value> operands) {
  IVLOG(5, "GatherOp::getResultType>")
  if (operands.size() != 2) {
    throw std::runtime_error("GatherOp requires 2 operands");
  }
  auto tensor = operands[0];
  auto tensorType = eltwise::getRankedTensorType(tensor.getType());
  auto tensorElementType = tensorType.getElementType();
  if (!tensorType.getRank()) {
    throw std::runtime_error(
        "'gather' requires first operand to have at least one dimension.");
  }
  auto index = operands[1];
  auto indexType = eltwise::getRankedTensorType(index.getType());
  auto indexElementType = indexType.getElementType();
  if (!indexElementType.isSignedInteger(32)) {
    throw std::runtime_error(
        "'gather' requires the data type for the second argument to be i32.");
  }
  SmallVector<int64_t, 4> shape;
  auto tensorShape = tensorType.getShape();
  auto indexShape = indexType.getShape();
  for (size_t i = 0; i < indexShape.size(); i++) {
    shape.push_back(indexShape[i]);
  }
  for (size_t i = 1; i < tensorShape.size(); i++) {
    shape.push_back(tensorShape[i]);
  }
  auto resultType = RankedTensorType::get(shape, tensorElementType);
  IVLOG(6, "  resultType: " << mlir::debugString(resultType));
  return resultType;
}

// ---- IndexOp ----

struct IndexCanonicalizer : public OpRewritePattern<IndexOp> {
  using OpRewritePattern<IndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexOp indexOp,
                                PatternRewriter &rewriter) const override {
    IVLOG(5, "IndexCanonicalizer::matchAndRewrite> "
                 << mlir::debugString(indexOp));
    auto op = indexOp.getOperation();
    SmallVector<Value, 4> operands(op->getOperands());
    auto resultType = IndexOp::getResultType(operands);
    if (resultType == indexOp.result().getType()) {
      return failure();
    }
    auto newOp = rewriter.create<IndexOp>(op->getLoc(), resultType,
                                          indexOp.axisAttr(), indexOp.dims());
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
    return success();
  }
};

void IndexOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<IndexCanonicalizer>(context);
}

Type IndexOp::getResultType(ArrayRef<Value> operands) {
  if (operands.size() < 1) {
    throw std::runtime_error("IndexOp requires at least one operand");
  }

  auto *context = operands.front().getContext();
  auto elementType = IntegerType::get(32, IntegerType::Signed, context);
  auto shape = eltwise::getShapeFromOperands(operands);
  return RankedTensorType::get(shape, elementType);
}

// ---- PrngOp ----

struct PrngCanonicalizer : public OpRewritePattern<PrngOp> {
  using OpRewritePattern<PrngOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PrngOp prngOp,
                                PatternRewriter &rewriter) const override {
    IVLOG(5,
          "PrngCanonicalizer::matchAndRewrite> " << mlir::debugString(prngOp));
    auto op = prngOp.getOperation();
    SmallVector<Value, 5> operands(op->getOperands());
    auto resultType = PrngOp::getResultType(operands);
    if (resultType == prngOp.result().getType()) {
      return failure();
    }
    auto stateType = prngOp.new_state().getType();
    SmallVector<Value, 4> dims(prngOp.dims());
    auto newOp = rewriter.create<PrngOp>(op->getLoc(), resultType, stateType,
                                         prngOp.state(), dims);
    rewriter.replaceOp(op, {newOp.result(), newOp.new_state()});
    util::UpdateFuncOpType(newOp.getOperation());
    return success();
  }
};

void PrngOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  results.insert<PrngCanonicalizer>(context);
}

Type PrngOp::getResultType(ArrayRef<Value> operands) {
  IVLOG(5, "PrngOp::getResultType>")
  if (operands.size() < 1) {
    throw std::runtime_error("PrngOp requires at least one operand");
  }
  auto state = operands.front();
  auto dims = operands.drop_front();
  auto shape = eltwise::getShapeFromOperands(dims);
  auto elementType = FloatType::getF32(state.getContext());
  return RankedTensorType::get(shape, elementType);
}

// ---- ReshapeOp ----

OpFoldResult ReshapeOp::fold(ArrayRef<Attribute> operands) {
  IVLOG(5, "ReshapeOp::fold");
  // reshape(x, x.shape) -> x
  if (tensor().getType() == getType()) {
    return tensor();
  }
  return {};
}

struct ReshapeCanonicalizer : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    IVLOG(5, "ReshapeCanonicalizer::matchAndRewrite> "
                 << mlir::debugString(reshapeOp));
    auto op = reshapeOp.getOperation();
    SmallVector<Value, 5> operands(op->getOperands());
    auto resultType = ReshapeOp::getResultType(operands);
    if (resultType == reshapeOp.result().getType()) {
      return failure();
    }
    SmallVector<Value, 4> dims(reshapeOp.dims());
    auto newOp = rewriter.create<ReshapeOp>(op->getLoc(), resultType,
                                            reshapeOp.tensor(), dims);
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
    return success();
  }
};

void ReshapeOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<ReshapeCanonicalizer>(context);
}

Type ReshapeOp::getResultType(ArrayRef<Value> operands) {
  IVLOG(5, "ReshapeOp::getResultType>")
  if (operands.empty()) {
    throw std::runtime_error("ReshapeOp requires at least 1 operand");
  }
  auto tensor = operands.front();
  auto dims = operands.drop_front();
  auto tensorType = eltwise::getRankedTensorType(tensor.getType());
  auto elementType = tensorType.getElementType();
  auto shape = eltwise::getShapeFromOperands(dims);
  return RankedTensorType::get(shape, elementType);
}

// --- ScatterOp ---

struct ScatterCanonicalizer : public OpRewritePattern<ScatterOp> {
  using OpRewritePattern<ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScatterOp scatterOp,
                                PatternRewriter &rewriter) const override {
    IVLOG(5, "IndexCanonicalizer::matchAndRewrite> "
                 << mlir::debugString(scatterOp));
    auto op = scatterOp.getOperation();
    SmallVector<Value, 3> operands(op->getOperands());
    auto resultType = ScatterOp::getResultType(operands);
    if (resultType == scatterOp.result().getType()) {
      return failure();
    }
    auto newOp =
        rewriter.create<ScatterOp>(op->getLoc(), resultType, scatterOp.tensor(),
                                   scatterOp.dims(), scatterOp.other());
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
    return success();
  }
};

void ScatterOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<ScatterCanonicalizer>(context);
}

Type ScatterOp::getResultType(ArrayRef<Value> operands) {
  IVLOG(5, "ScatterOp::getResultType>")
  if (operands.size() != 3) {
    throw std::runtime_error("ScatterOp requires 3 operands");
  }
  auto tensor = operands[0];
  auto tensorType = eltwise::getRankedTensorType(tensor.getType());
  auto tensorElementType = tensorType.getElementType();
  const auto &tensorShape = tensorType.getShape();
  if (!tensorType.getRank()) {
    throw std::runtime_error(
        "'scatter' requires first operand to have at least one dimension.");
  }
  auto index = operands[1];
  auto indexType = eltwise::getRankedTensorType(index.getType());
  auto indexElementType = indexType.getElementType();
  if (!indexElementType.isSignedInteger(32)) {
    throw std::runtime_error(
        "'scatter' requires the data type for the second argument to be i32.");
  }
  auto other = operands[2];
  auto otherType = eltwise::getRankedTensorType(other.getType());
  const auto &otherShape = otherType.getShape();
  SmallVector<int64_t, 4> shape{otherShape[0]};
  for (unsigned i = indexType.getRank(); i < tensorType.getRank(); i++) {
    shape.emplace_back(tensorShape[i]);
  }
  auto resultType = RankedTensorType::get(shape, tensorElementType);
  IVLOG(6, "  resultType: " << mlir::debugString(resultType));
  return resultType;
}

// ---- ShapeOp ----

struct ShapeCanonicalizer : public OpRewritePattern<ShapeOp> {
  using OpRewritePattern<ShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ShapeOp shapeOp,
                                PatternRewriter &rewriter) const override {
    IVLOG(5, "ShapeCanonicalizer::matchAndRewrite> "
                 << mlir::debugString(shapeOp));
    auto op = shapeOp.getOperation();
    SmallVector<Value, 1> operands(op->getOperands());
    auto resultType = ShapeOp::getResultType(operands);
    if (resultType == shapeOp.result().getType()) {
      return failure();
    }
    auto newOp =
        rewriter.create<ShapeOp>(op->getLoc(), resultType, shapeOp.tensor());
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
    return success();
  }
};

void ShapeOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<ShapeCanonicalizer>(context);
}

Type ShapeOp::getResultType(ArrayRef<Value> operands) {
  IVLOG(5, "ShapeOp::getResultType>")
  if (operands.size() != 1) {
    throw std::runtime_error("ShapeOp requires 1 operand");
  }
  auto tensor = operands[0];
  auto tensorType = eltwise::getRankedTensorType(tensor.getType());
  auto elementType = IntegerType::get(32, IntegerType::Signed,
                                      tensor.getContext()); // TODO: index type?
  return RankedTensorType::get({tensorType.getRank()}, elementType);
}

// ---- ConstantOp ----

void printConstantOp(OpAsmPrinter *printer, ConstantOp op) {
  *printer << op.getOperation()->getName() << ' ' << op.value().getZExtValue();
}

ParseResult parseConstantOp(OpAsmParser *parser, OperationState &result) {
  auto indexType = parser->getBuilder().getIndexType();
  result.addTypes(indexType);
  IntegerAttr value;
  return parser->parseAttribute(value, indexType, "value", result.attributes);
}

LogicalResult verifyConstantOp(ConstantOp op) { return success(); }

// ---- ContractionOp ----

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

// ---- TraceOp ----

struct TraceOpCanonicalizer : public OpRewritePattern<TraceOp> {
  using OpRewritePattern<TraceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TraceOp op,
                                PatternRewriter &rewriter) const override {
    IVLOG(5,
          "TraceOpCanonicalizer::matchAndRewrite> " << mlir::debugString(op));
    if (op.in().getType() == op.out().getType()) {
      return failure();
    }
    auto newOp = rewriter.create<TraceOp>(op.getLoc(), op.in(), op.msg());
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
    return success();
  }
};

void TraceOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<TraceOpCanonicalizer>(context);
}

#define GET_OP_CLASSES
#include "pmlc/dialect/tile/ir/ops.cc.inc"

} // namespace pmlc::dialect::tile
