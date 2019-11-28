// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/ops.h"

#include <vector>

#include "llvm/ADT/SetVector.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/DebugStringHelper.h"

#include "base/util/logging.h"
#include "pmlc/dialect/eltwise/util.h"
#include "pmlc/util/util.h"

namespace pmlc::dialect::tile {

using eltwise::constFoldBinaryOp;
using eltwise::constFoldUnaryOp;
using eltwise::m_One;
using eltwise::m_Zero;
using llvm::SetVector;
using llvm::SmallVector;
using mlir::AffineExpr;
using mlir::ArrayAttr;
using mlir::failure;
using mlir::FloatAttr;
using mlir::IntegerAttr;
using mlir::OpRewritePattern;
using mlir::PatternMatchResult;
using mlir::PatternRewriter;
using mlir::success;
using mlir::Value;

OpFoldResult AffineConstantOp::fold(ArrayRef<Attribute> operands) {
  IVLOG(5, "AffineConstantOp::fold");
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

OpFoldResult AffineAddOp::fold(ArrayRef<Attribute> operands) {
  IVLOG(5, "AffineAddOp::fold");
  /// add(x, 0) -> x
  if (matchPattern(rhs(), m_Zero())) {
    return lhs();
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a + b; });
}

OpFoldResult AffineDivOp::fold(ArrayRef<Attribute> operands) {
  IVLOG(5, "AffineDivOp::fold");
  // Don't fold if it requires division by zero.
  if (matchPattern(rhs(), m_Zero())) {
    return {};
  }
  // div(x, 1) -> x
  if (matchPattern(rhs(), m_One())) {
    return lhs();
  }
  // div(0, x) -> 0
  if (matchPattern(lhs(), m_Zero())) {
    Builder builder(getContext());
    return builder.getZeroAttr(builder.getIntegerType(64));
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a / b; });
}

OpFoldResult AffineMulOp::fold(ArrayRef<Attribute> operands) {
  IVLOG(5, "AffineMulOp::fold");
  // mul(x, 0) -> 0
  if (matchPattern(rhs(), m_Zero())) {
    IVLOG(5, "mul(x, 0) -> 0");
    return rhs();
  }
  // mul(x, 1) -> x
  if (matchPattern(rhs(), m_One())) {
    IVLOG(5, "mul(x, 1) -> x");
    return lhs();
  }
  return constFoldBinaryOp(operands, [](double a, double b) {
    IVLOG(5, a << " * " << b << " = " << a * b);
    return a * b;
  });
}

OpFoldResult AffineNegOp::fold(ArrayRef<Attribute> operands) {
  IVLOG(5, "AffineNegOp::fold");
  return constFoldUnaryOp(operands, [](double x) { return -x; });
}

OpFoldResult AffineMaxOp::fold(ArrayRef<Attribute> operands) {
  IVLOG(5, "AffineMaxOp::fold");
  return constFoldBinaryOp(operands, [](double a, double b) { return fmax(a, b); });
}

OpFoldResult AffineMinOp::fold(ArrayRef<Attribute> operands) {
  IVLOG(5, "AffineMinOp::fold");
  return constFoldBinaryOp(operands, [](double a, double b) { return fmin(a, b); });
}

OpFoldResult AffineSubOp::fold(ArrayRef<Attribute> operands) {
  IVLOG(5, "AffineSubOp::fold");
  // sub(x, x) -> 0
  if (lhs() == rhs()) {
    IVLOG(5, "sub(x, x) -> 0");
    Builder builder(getContext());
    return builder.getZeroAttr(builder.getIntegerType(64));
  }
  /// sub(x, 0) -> x
  if (matchPattern(rhs(), m_Zero())) {
    IVLOG(5, "sub(x, 0) -> x");
    return lhs();
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a - b; });
}

OpFoldResult DimOp::fold(ArrayRef<Attribute> operands) {
  IVLOG(5, "DimOp::fold");
  auto type = tensor()->getType().dyn_cast<mlir::TensorType>();
  if (!type) {
    return {};
  }
  auto size = type.getDimSize(dim().getSExtValue());
  if (mlir::ShapedType::isDynamic(size)) {
    return {};
  }
  return IntegerAttr::get(mlir::IntegerType::get(64, getContext()), size);
}

struct AffineDomainFolder : public OpRewritePattern<AffineDomainOp> {
  using OpRewritePattern<AffineDomainOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(AffineDomainOp op, PatternRewriter& rewriter) const override {
    IVLOG(5, "AffineDomainFolder::matchAndRewrite>");
    auto terminator = op.body().front().getTerminator();
    while (!llvm::isa<ContractionOp>(terminator)) {
      terminator = terminator->getRegion(0).front().getTerminator();
    }
    auto contractionOp = llvm::cast<ContractionOp>(terminator);
    auto sizeMapOp = llvm::dyn_cast<AffineSizeMapOp>(contractionOp.getSizeMap()->getDefiningOp());
    if (!sizeMapOp) {
      return matchFailure();
    }
    SmallVector<Value*, 4> sizes(sizeMapOp.sizes());
    auto shape = eltwise::ComputeShape(sizes);
    auto sourceType = op.getType().cast<RankedTensorType>();
    auto targetType = RankedTensorType::get(shape, sourceType.getElementType());
    IVLOG(6, "  sourceType: " << mlir::debugString(sourceType));
    IVLOG(6, "  targetType: " << mlir::debugString(targetType));
    if (sourceType == targetType) {
      return matchFailure();
    }
    BoolAttr no_reduce;
    if (auto optional = op.no_reduce()) {
      no_reduce = rewriter.getBoolAttr(*optional);
    }
    auto newOp = rewriter.create<AffineDomainOp>(op.getLoc(), targetType, no_reduce);
    if (auto attr = op.getAttrOfType<StringAttr>("name")) {
      newOp.setAttr("name", attr);
    }
    if (auto attr = op.getAttrOfType<ArrayAttr>("idx_names")) {
      newOp.setAttr("idx_names", attr);
    }
    newOp.body().takeBody(op.body());
    rewriter.replaceOp(op, {newOp.result()});
    util::UpdateFuncOpType(newOp.getOperation());
    return matchSuccess();
  }
};

void AffineDomainOp::getCanonicalizationPatterns(  //
    OwningRewritePatternList& results,             //
    MLIRContext* context) {
  results.insert<AffineDomainFolder>(context);
}

struct ContractionBuilder {
  MLIRContext* context;
  SetVector<Value*> idxs;
  std::vector<Value*> tensors;
  std::vector<AffineExpr> sink;
  std::vector<std::vector<AffineExpr>> srcs;
  std::vector<AffineExpr> cons;

  ContractionBuilder(AffineMapOp sink, AffineConstraintsOp constraints)
      : context(sink.getContext()), sink(addDims(sink.dims())) {
    SmallVector<Value*, 8> pairs(constraints.pairs());
    for (auto it = pairs.begin(); it != pairs.end(); it++) {
      auto lhs = *it++;
      auto rhs = *it;
      cons.emplace_back(makeConstraint(lhs, rhs));
    }
  }

  std::vector<AffineExpr> addDims(Operation::operand_range dims) {
    std::vector<AffineExpr> exprs;
    for (auto dim : dims) {
      exprs.emplace_back(makeExpr(dim));
    }
    return exprs;
  }

  void addSourceMap(AffineTensorMapOp op) {
    tensors.emplace_back(op.tensor());
    srcs.emplace_back(addDims(op.dims()));
  }

  AffineMap makeMap(ArrayRef<AffineExpr> exprs) {
    if (exprs.size()) {
      return AffineMap::get(idxs.size(), 0, exprs);
    }
    return AffineMap::get(context);
  }

  std::vector<AffineMap> getSources() {
    std::vector<AffineMap> ret;
    for (auto src : srcs) {
      ret.emplace_back(makeMap(src));
    }
    return ret;
  }

  AffineMap getSink() {  //
    return makeMap(sink);
  }

  IntegerSet getConstraints() {
    if (cons.empty()) {
      return IntegerSet::getEmptySet(idxs.size(), 0, context);
    }
    SmallVector<bool, 4> flags(cons.size(), false);
    return IntegerSet::get(idxs.size(), 0, cons, flags);
  }

  AffineExpr makeConstraint(Value* lhs, Value* rhs) {  //
    return makeExpr(lhs) - makeExpr(rhs);
  }

  AffineExpr makeExpr(Value* value) {
    IVLOG(3, "MakePoly: " << mlir::debugString(*value));
    auto defOp = value->getDefiningOp();
    if (auto op = llvm::dyn_cast<AffineIndexOp>(defOp)) {
      idxs.insert(value);
      auto pos = std::distance(idxs.begin(), llvm::find(idxs, value));
      return mlir::getAffineDimExpr(pos, context);
    }
    if (auto op = llvm::dyn_cast<AffineConstantOp>(defOp)) {
      return mlir::getAffineConstantExpr(op.value().getSExtValue(), context);
    }
    if (auto op = llvm::dyn_cast<AffineAddOp>(defOp)) {
      return makeExpr(op.lhs()) + makeExpr(op.rhs());
    }
    if (auto op = llvm::dyn_cast<AffineMulOp>(defOp)) {
      return makeExpr(op.lhs()) * makeExpr(op.rhs());
    }
    if (auto op = llvm::dyn_cast<AffineDivOp>(defOp)) {
      return makeExpr(op.lhs()).floorDiv(makeExpr(op.rhs()));
    }
    if (auto op = llvm::dyn_cast<AffineNegOp>(defOp)) {
      return -makeExpr(op.input());
    }
    if (auto op = llvm::dyn_cast<AffineSubOp>(defOp)) {
      return makeExpr(op.lhs()) - makeExpr(op.rhs());
    }
    throw std::runtime_error("Invalid affine op");
  }
};

struct SymbolicContractionCanonicalizer : OpRewritePattern<AffineSymbolicContractionOp> {
  using OpRewritePattern<AffineSymbolicContractionOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(AffineSymbolicContractionOp op, PatternRewriter& rewriter) const override {
    auto sizeMapOp = llvm::cast<AffineMapOp>(op.size()->getDefiningOp());
    SmallVector<Value*, 4> sizeDims(sizeMapOp.dims());
    auto shape = eltwise::ComputeShape(sizeDims);
    auto sourceType = op.result()->getType().cast<RankedTensorType>();
    auto resultType = RankedTensorType::get(shape, sourceType.getElementType());
    if (!resultType.hasStaticShape()) {
      return matchFailure();
    }

    auto sinkMapOp = llvm::cast<AffineMapOp>(op.sink()->getDefiningOp());
    auto consOp = llvm::cast<AffineConstraintsOp>(op.cons()->getDefiningOp());
    ContractionBuilder builder(sinkMapOp, consOp);
    for (auto src : op.srcs()) {
      auto mapOp = llvm::cast<AffineTensorMapOp>(src->getDefiningOp());
      builder.addSourceMap(mapOp);
    }

    auto newOp = rewriter.create<AffineContractionOp>(  //
        op.getLoc(),                                    //
        resultType,                                     //
        op.init(),                                      //
        builder.tensors,                                //
        op.agg(),                                       //
        op.combo(),                                     //
        builder.getSink(),                              //
        builder.getSources(),                           //
        builder.getConstraints(),                       //
        op.no_reduce().hasValue());
    rewriter.replaceOp(op, newOp.result());

    util::UpdateFuncOpType(newOp.getOperation());

    return matchSuccess();
  }
};

void AffineSymbolicContractionOp::getCanonicalizationPatterns(  //
    OwningRewritePatternList& results,                          //
    MLIRContext* context) {
  results.insert<SymbolicContractionCanonicalizer>(context);
}

void AffineContractionOp::build(  //
    Builder* builder,             //
    OperationState& result,       //
    Type resultType,              //
    Value* init,                  //
    ArrayRef<Value*> tensors,     //
    AggregationKind agg,          //
    CombinationKind combo,        //
    AffineMap sink,               //
    ArrayRef<AffineMap> srcs,     //
    IntegerSet cons,              //
    bool no_reduce) {
  result.addOperands(init);
  result.addOperands(tensors);
  result.addTypes(resultType);
  result.addAttribute("agg", builder->getI64IntegerAttr(static_cast<int64_t>(agg)));
  result.addAttribute("combo", builder->getI64IntegerAttr(static_cast<int64_t>(combo)));
  result.addAttribute("sink", AffineMapAttr::get(sink));
  result.addAttribute("srcs", builder->getAffineMapArrayAttr(srcs));
  result.addAttribute("cons", IntegerSetAttr::get(cons));
  if (no_reduce) {
    result.addAttribute("no_reduce", builder->getUnitAttr());
  }
}

//
// --- GatherOp ---
//

struct GatherCanonicalizer : public OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(GatherOp gatherOp, PatternRewriter& rewriter) const override {
    IVLOG(5, "IndexCanonicalizer::matchAndRewrite> " << mlir::debugString(gatherOp));
    auto op = gatherOp.getOperation();
    SmallVector<Value*, 2> operands(op->getOperands());
    auto resultType = GatherOp::getResultType(operands);
    if (resultType == gatherOp.result()->getType()) {
      return Pattern::matchFailure();
    }
    auto newOp = rewriter.create<GatherOp>(op->getLoc(), resultType, gatherOp.tensor(), gatherOp.dims());
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
    return Pattern::matchSuccess();
  }
};

void GatherOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<GatherCanonicalizer>(context);
}

Type GatherOp::getResultType(ArrayRef<Value*> operands) {
  IVLOG(5, "GatherOp::getResultType>")
  if (operands.size() != 2) {
    throw std::runtime_error("GatherOp requires 2 operands");
  }
  auto tensor = operands[0];
  auto tensorType = eltwise::getRankedTensorType(tensor->getType());
  auto tensorElementType = tensorType.getElementType();
  if (!tensorType.getRank()) {
    throw std::runtime_error("'gather' requires first operand to have at least one dimension.");
  }
  auto index = operands[1];
  auto indexType = eltwise::getRankedTensorType(index->getType());
  auto indexElementType = indexType.getElementType().dyn_cast<ScalarType>();
  if (!indexElementType || indexElementType.type() != eltwise::DataType::INT32) {
    throw std::runtime_error("'gather' requires the data type for the second argument to be INT32.");
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

//
// ---- IndexOp ----
//

struct IndexCanonicalizer : public OpRewritePattern<IndexOp> {
  using OpRewritePattern<IndexOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(IndexOp indexOp, PatternRewriter& rewriter) const override {
    IVLOG(5, "IndexCanonicalizer::matchAndRewrite> " << mlir::debugString(indexOp));
    auto op = indexOp.getOperation();
    SmallVector<Value*, 2> operands(op->getOperands());
    auto resultType = IndexOp::getResultType(operands);
    if (resultType == indexOp.result()->getType()) {
      return Pattern::matchFailure();
    }
    auto dim = indexOp.getAttrOfType<IntegerAttr>("dim");
    auto newOp = rewriter.create<IndexOp>(op->getLoc(), resultType, indexOp.tensor(), dim);
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
    return Pattern::matchSuccess();
  }
};

void IndexOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<IndexCanonicalizer>(context);
}

Type IndexOp::getResultType(ArrayRef<Value*> operands) {
  IVLOG(5, "IndexOp::getResultType>")
  for (auto operand : operands) {
    IVLOG(6, "  operand: " << mlir::debugString(*operand));
  }
  if (operands.size() != 1) {
    throw std::runtime_error("IndexOp requires 1 operand");
  }
  auto tensor = operands.front();
  auto tensorType = eltwise::getRankedTensorType(tensor->getType());
  auto elementType = ScalarType::get(tensor->getContext(), eltwise::DataType::INT32);
  IVLOG(6, "  elementType: " << mlir::debugString(elementType));
  auto resultType = RankedTensorType::get(tensorType.getShape(), elementType);
  IVLOG(6, "  resultType: " << mlir::debugString(resultType));
  return resultType;
}

//
// ---- PrngOp ----
//

struct PrngCanonicalizer : public OpRewritePattern<PrngOp> {
  using OpRewritePattern<PrngOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(PrngOp prngOp, PatternRewriter& rewriter) const override {
    IVLOG(5, "PrngCanonicalizer::matchAndRewrite> " << mlir::debugString(prngOp));
    auto op = prngOp.getOperation();
    SmallVector<Value*, 5> operands(op->getOperands());
    auto resultType = PrngOp::getResultType(operands);
    if (resultType == prngOp.result()->getType()) {
      return Pattern::matchFailure();
    }
    auto stateType = prngOp.new_state()->getType();
    SmallVector<Value*, 4> dims(prngOp.dims());
    auto newOp = rewriter.create<PrngOp>(op->getLoc(), resultType, stateType, prngOp.state(), dims);
    rewriter.replaceOp(op, {newOp.result(), newOp.new_state()});
    util::UpdateFuncOpType(newOp.getOperation());
    return Pattern::matchSuccess();
  }
};

void PrngOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<PrngCanonicalizer>(context);
}

Type PrngOp::getResultType(ArrayRef<Value*> operands) {
  IVLOG(5, "PrngOp::getResultType>")
  if (operands.size() < 1) {
    throw std::runtime_error("PrngOp requires at least one operand");
  }
  auto state = operands.front();
  auto dims = operands.drop_front();
  auto shape = eltwise::ComputeShape(dims);
  auto elementType = ScalarType::get(state->getContext(), DataType::FLOAT32);
  return RankedTensorType::get(shape, elementType);
}

//
// ---- ReshapeOp ----
//

struct ReshapeCanonicalizer : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(ReshapeOp reshapeOp, PatternRewriter& rewriter) const override {
    IVLOG(5, "ReshapeCanonicalizer::matchAndRewrite> " << mlir::debugString(reshapeOp));
    auto op = reshapeOp.getOperation();
    SmallVector<Value*, 5> operands(op->getOperands());
    auto resultType = ReshapeOp::getResultType(operands);
    if (resultType == reshapeOp.result()->getType()) {
      return Pattern::matchFailure();
    }
    SmallVector<Value*, 4> dims(reshapeOp.dims());
    auto newOp = rewriter.create<ReshapeOp>(op->getLoc(), resultType, reshapeOp.tensor(), dims);
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
    return Pattern::matchSuccess();
  }
};

void ReshapeOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<ReshapeCanonicalizer>(context);
}

Type ReshapeOp::getResultType(ArrayRef<Value*> operands) {
  IVLOG(5, "ReshapeOp::getResultType>")
  if (operands.size() < 2) {
    throw std::runtime_error("ReshapeOp requires at least 2 operands");
  }
  auto tensor = operands.front();
  auto dims = operands.drop_front();
  auto tensorType = eltwise::getRankedTensorType(tensor->getType());
  auto elementType = tensorType.getElementType();
  auto shape = eltwise::ComputeShape(dims);
  return RankedTensorType::get(shape, elementType);
}

//
// --- ScatterOp ---
//

struct ScatterCanonicalizer : public OpRewritePattern<ScatterOp> {
  using OpRewritePattern<ScatterOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(ScatterOp scatterOp, PatternRewriter& rewriter) const override {
    IVLOG(5, "IndexCanonicalizer::matchAndRewrite> " << mlir::debugString(scatterOp));
    auto op = scatterOp.getOperation();
    SmallVector<Value*, 3> operands(op->getOperands());
    auto resultType = ScatterOp::getResultType(operands);
    if (resultType == scatterOp.result()->getType()) {
      return Pattern::matchFailure();
    }
    auto newOp =
        rewriter.create<ScatterOp>(op->getLoc(), resultType, scatterOp.tensor(), scatterOp.dims(), scatterOp.other());
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
    return Pattern::matchSuccess();
  }
};

void ScatterOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<ScatterCanonicalizer>(context);
}

Type ScatterOp::getResultType(ArrayRef<Value*> operands) {
  IVLOG(5, "ScatterOp::getResultType>")
  if (operands.size() != 3) {
    throw std::runtime_error("ScatterOp requires 3 operands");
  }
  auto tensor = operands[0];
  auto tensorType = eltwise::getRankedTensorType(tensor->getType());
  auto tensorElementType = tensorType.getElementType();
  const auto& tensorShape = tensorType.getShape();
  if (!tensorType.getRank()) {
    throw std::runtime_error("'scatter' requires first operand to have at least one dimension.");
  }
  auto index = operands[1];
  auto indexType = eltwise::getRankedTensorType(index->getType());
  auto indexElementType = indexType.getElementType().dyn_cast<ScalarType>();
  if (!indexElementType || indexElementType.type() != eltwise::DataType::INT32) {
    throw std::runtime_error("'scatter' requires the data type for the second argument to be INT32.");
  }
  auto other = operands[2];
  auto otherType = eltwise::getRankedTensorType(other->getType());
  const auto& otherShape = otherType.getShape();
  SmallVector<int64_t, 4> shape{otherShape[0]};
  for (unsigned i = indexType.getRank(); i < tensorType.getRank(); i++) {
    shape.emplace_back(tensorShape[i]);
  }
  auto resultType = RankedTensorType::get(shape, tensorElementType);
  IVLOG(6, "  resultType: " << mlir::debugString(resultType));
  return resultType;
}

//
// ---- ShapeOp ----
//

struct ShapeCanonicalizer : public OpRewritePattern<ShapeOp> {
  using OpRewritePattern<ShapeOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(ShapeOp shapeOp, PatternRewriter& rewriter) const override {
    IVLOG(5, "ShapeCanonicalizer::matchAndRewrite> " << mlir::debugString(shapeOp));
    auto op = shapeOp.getOperation();
    SmallVector<Value*, 1> operands(op->getOperands());
    auto resultType = ShapeOp::getResultType(operands);
    if (resultType == shapeOp.result()->getType()) {
      return Pattern::matchFailure();
    }
    auto newOp = rewriter.create<ShapeOp>(op->getLoc(), resultType, shapeOp.tensor());
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
    return Pattern::matchSuccess();
  }
};

void ShapeOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<ShapeCanonicalizer>(context);
}

Type ShapeOp::getResultType(ArrayRef<Value*> operands) {
  IVLOG(5, "ShapeOp::getResultType>")
  if (operands.size() != 1) {
    throw std::runtime_error("ShapeOp requires 1 operand");
  }
  auto tensor = operands[0];
  auto tensorType = eltwise::getRankedTensorType(tensor->getType());
  auto elementType = ScalarType::get(tensor->getContext(), eltwise::DataType::INT32);  // TODO: index type?
  return RankedTensorType::get({tensorType.getRank()}, elementType);
}

// ---- DimOp ----

void printDimOp(OpAsmPrinter* printer, DimOp op) {  //
  *printer << op.getOperation()->getName() << ' ';
  printer->printOperand(op.tensor());
  *printer << '[' << op.dim().getZExtValue() << ']';
}

ParseResult parseDimOp(OpAsmParser* parser, OperationState& result) {
  auto indexType = parser->getBuilder().getIndexType();
  OpAsmParser::OperandType tensor;
  IntegerAttr dim;
  Type type;
  if (parser->parseOperand(tensor) ||                           //
      parser->parseLSquare() ||                                 //
      parser->parseAttribute(dim, "dim", result.attributes) ||  //
      parser->parseRSquare() ||                                 //
      parser->parseColonType(type) ||                           //
      parser->resolveOperand(tensor, type, result.operands)) {
    return failure();
  }
  result.addTypes(indexType);
  return success();
}

LogicalResult verifyDimOp(DimOp op) {  //
  return success();
}

// ---- AffineConstantOp ----

void printAffineConstantOp(OpAsmPrinter* printer, AffineConstantOp op) {  //
  *printer << op.getOperation()->getName() << ' ' << op.value().getZExtValue();
}

ParseResult parseAffineConstantOp(OpAsmParser* parser, OperationState& result) {
  IntegerAttr value;
  if (parser->parseAttribute(value, "value", result.attributes)) {
    return failure();
  }
  auto indexType = parser->getBuilder().getIndexType();
  result.addTypes(indexType);
  return success();
}

LogicalResult verifyAffineConstantOp(AffineConstantOp op) {  //
  return success();
}

// ---- AffineIndexOp ----

void printAffineIndexOp(OpAsmPrinter* printer, AffineIndexOp op) {  //
  *printer << op.getOperation()->getName();
}

ParseResult parseAffineIndexOp(OpAsmParser* parser, OperationState& result) {
  auto indexType = parser->getBuilder().getIndexType();
  result.addTypes(indexType);
  return success();
}

LogicalResult verifyAffineIndexOp(AffineIndexOp op) {  //
  return success();
}

// ---- AffineTensorMapOp ----

void printAffineTensorMapOp(OpAsmPrinter* printer, AffineTensorMapOp op) {
  *printer << op.getOperation()->getName() << ' ';
  printer->printOperand(op.tensor());
  *printer << '[';
  printer->printOperands(op.dims());
  *printer << ']';
}

ParseResult parseAffineTensorMapOp(OpAsmParser* parser, OperationState& result) {
  auto indexType = parser->getBuilder().getIndexType();
  OpAsmParser::OperandType tensor;
  SmallVector<OpAsmParser::OperandType, 4> dims;
  Type type;
  if (parser->parseOperand(tensor) ||                                    //
      parser->parseOperandList(dims, OpAsmParser::Delimiter::Square) ||  //
      parser->parseColonType(type) ||                                    //
      parser->resolveOperand(tensor, type, result.operands) ||           //
      parser->resolveOperands(dims, indexType, result.operands)) {
    return failure();
  }
  result.addTypes(AffineTensorMapType::get(result.getContext()));
  return success();
}

LogicalResult verifyAffineTensorMapOp(AffineTensorMapOp op) {  //
  return success();
}

// ---- AffineMapOp ----

void printAffineMapOp(OpAsmPrinter* printer, AffineMapOp op) {
  *printer << op.getOperation()->getName() << ' ';
  printer->printOperands(op.dims());
}

ParseResult parseAffineMapOp(OpAsmParser* parser, OperationState& result) {
  auto indexType = parser->getBuilder().getIndexType();
  SmallVector<OpAsmParser::OperandType, 4> dims;
  Type type;
  if (parser->parseOperandList(dims) ||  //
      parser->resolveOperands(dims, indexType, result.operands)) {
    return failure();
  }
  result.addTypes(AffineMapType::get(result.getContext()));
  return success();
}

LogicalResult verifyAffineMapOp(AffineMapOp op) {  //
  return success();
}

// ---- AffineConstraintsOp ----

void printAffineConstraintsOp(OpAsmPrinter* printer, AffineConstraintsOp op) {
  *printer << op.getOperation()->getName() << ' ';
  *printer << '(';
  printer->printOperands(op.pairs());
  *printer << ')';
}

ParseResult parseAffineConstraintsOp(OpAsmParser* parser, OperationState& result) {
  auto indexType = parser->getBuilder().getIndexType();
  SmallVector<OpAsmParser::OperandType, 4> dims;
  Type type;
  if (parser->parseOperandList(dims, OpAsmParser::Delimiter::Paren) ||  //
      parser->resolveOperands(dims, indexType, result.operands)) {
    return failure();
  }
  result.addTypes(AffineConstraintsType::get(result.getContext()));
  return success();
}

LogicalResult verifyAffineConstraintsOp(AffineConstraintsOp op) {  //
  return success();
}

// ---- AffineSymbolicContractionOp ----

void printAffineSymbolicContractionOp(OpAsmPrinter* printer, AffineSymbolicContractionOp op) {  //
  *printer << op.getOperation()->getName() << ' ';
  *printer << util::stringifyAggregationKind(op.agg());
  *printer << ", ";
  *printer << util::stringifyCombinationKind(op.combo());
  *printer << ", ";
  printer->printOperand(op.init());
  *printer << ", ";
  printer->printOperand(op.cons());
  *printer << ", ";
  printer->printOperand(op.size());
  *printer << ", ";
  printer->printOperand(op.sink());
  *printer << ", ";
  printer->printOperands(op.srcs());
  *printer << " : ";
  printer->printType(op.init()->getType());
  *printer << " -> ";
  printer->printType(op.result()->getType());
}

ParseResult parseAffineSymbolicContractionOp(OpAsmParser* parser, OperationState& result) {
  StringRef strAgg;
  StringRef strCombo;
  OpAsmParser::OperandType init;
  Type initType;
  OpAsmParser::OperandType cons;
  auto consType = AffineConstraintsType::get(result.getContext());
  OpAsmParser::OperandType size;
  auto mapType = AffineMapType::get(result.getContext());
  OpAsmParser::OperandType sink;
  auto tmapType = AffineTensorMapType::get(result.getContext());
  SmallVector<OpAsmParser::OperandType, 3> srcs;
  Type resultType;
  if (parser->parseKeyword(&strAgg) ||     //
      parser->parseComma() ||              //
      parser->parseKeyword(&strCombo) ||   //
      parser->parseComma() ||              //
      parser->parseOperand(init) ||        //
      parser->parseComma() ||              //
      parser->parseOperand(cons) ||        //
      parser->parseComma() ||              //
      parser->parseOperand(size) ||        //
      parser->parseComma() ||              //
      parser->parseOperand(sink) ||        //
      parser->parseComma() ||              //
      parser->parseOperandList(srcs) ||    //
      parser->parseColonType(initType) ||  //
      parser->parseArrow() ||              //
      parser->parseType(resultType)) {
    return failure();
  }

  if (parser->resolveOperand(init, initType, result.operands) ||
      parser->resolveOperand(cons, consType, result.operands) ||
      parser->resolveOperand(size, mapType, result.operands) ||
      parser->resolveOperand(sink, mapType, result.operands) ||
      parser->resolveOperands(srcs, tmapType, result.operands)) {
    return failure();
  }

  auto agg = util::symbolizeAggregationKind(strAgg);
  if (!agg) {
    failure();
  }
  result.addAttribute("agg", parser->getBuilder().getI64IntegerAttr(static_cast<int64_t>(agg.getValue())));

  auto combo = util::symbolizeCombinationKind(strCombo);
  if (!combo) {
    failure();
  }
  result.addAttribute("combo", parser->getBuilder().getI64IntegerAttr(static_cast<int64_t>(combo.getValue())));

  result.addTypes(resultType);
  return success();
}

LogicalResult verifyAffineSymbolicContractionOp(AffineSymbolicContractionOp op) {  //
  return success();
}

// ---- AffineContractionOp ----

void printAffineContractionOp(OpAsmPrinter* printer, AffineContractionOp op) {  //
  *printer << op.getOperation()->getName() << ' ';
  *printer << util::stringifyAggregationKind(op.agg());
  *printer << ", ";
  *printer << util::stringifyCombinationKind(op.combo());
  *printer << ", ";
  printer->printOperand(op.init());
  *printer << ", ";
  printer->printOperands(op.tensors());
  *printer << ' ';
  printer->printOptionalAttrDict(op.getAttrs(), {"agg", "combo"});
  *printer << " : ";
  printer->printType(op.init()->getType());
  *printer << ", ";
  SmallVector<Value*, 4> tensors(op.tensors());
  mlir::interleaveComma(op.tensors(), printer->getStream(),
                        [&](Value* operand) { printer->printType(operand->getType()); });
  *printer << " -> ";
  printer->printType(op.result()->getType());
}

ParseResult parseAffineContractionOp(OpAsmParser* parser, OperationState& result) {
  StringRef strAgg;
  StringRef strCombo;
  OpAsmParser::OperandType init;
  SmallVector<OpAsmParser::OperandType, 3> tensors;
  SmallVector<Type, 4> types;
  Type resultType;
  if (parser->parseKeyword(&strAgg) ||                     //
      parser->parseComma() ||                              //
      parser->parseKeyword(&strCombo) ||                   //
      parser->parseComma() ||                              //
      parser->parseOperand(init) ||                        //
      parser->parseComma() ||                              //
      parser->parseOperandList(tensors) ||                 //
      parser->parseOptionalAttrDict(result.attributes) ||  //
      parser->parseColonTypeList(types) ||                 //
      parser->parseArrow() ||                              //
      parser->parseType(resultType)) {
    return failure();
  }

  auto loc = parser->getCurrentLocation();
  auto tensorTypes = llvm::makeArrayRef(types).drop_front();
  if (parser->resolveOperand(init, types.front(), result.operands) ||
      parser->resolveOperands(tensors, tensorTypes, loc, result.operands)) {
    return failure();
  }

  auto agg = util::symbolizeAggregationKind(strAgg);
  if (!agg) {
    failure();
  }
  result.addAttribute("agg", parser->getBuilder().getI64IntegerAttr(static_cast<int64_t>(agg.getValue())));

  auto combo = util::symbolizeCombinationKind(strCombo);
  if (!combo) {
    failure();
  }
  result.addAttribute("combo", parser->getBuilder().getI64IntegerAttr(static_cast<int64_t>(combo.getValue())));

  result.addTypes(resultType);
  return success();
}

LogicalResult verifyAffineContractionOp(AffineContractionOp op) {  //
  return success();
}

// ---- AffineBinaryOp ----

template <typename AffineBinaryOp>
void printAffineBinaryOp(OpAsmPrinter* printer, AffineBinaryOp op) {
  *printer << op.getOperation()->getName() << ' ';
  printer->printOperands(op.getOperands());
}

ParseResult parseAffineBinaryOp(OpAsmParser* parser, OperationState& result) {
  auto indexType = parser->getBuilder().getIndexType();
  OpAsmParser::OperandType lhs;
  OpAsmParser::OperandType rhs;
  if (parser->parseOperand(lhs) ||                                //
      parser->parseComma() ||                                     //
      parser->parseOperand(rhs) ||                                //
      parser->resolveOperand(lhs, indexType, result.operands) ||  //
      parser->resolveOperand(rhs, indexType, result.operands)) {
    return failure();
  }
  result.addTypes(IndexType::get(result.getContext()));
  return success();
}

#include "pmlc/dialect/tile/interfaces.cc.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/tile/ops.cc.inc"

}  // namespace pmlc::dialect::tile
