// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/ops.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
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
using llvm::SmallVector;
using mlir::ArrayAttr;
using mlir::FloatAttr;
using mlir::IntegerAttr;
using mlir::OpRewritePattern;
using mlir::PatternMatchResult;
using mlir::PatternRewriter;
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

namespace {

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
    llvm::SmallVector<Value*, 4> sizes(sizeMapOp.sizes());
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

}  // namespace

void AffineDomainOp::getCanonicalizationPatterns(  //
    OwningRewritePatternList& results,             //
    MLIRContext* context) {
  results.insert<AffineDomainFolder>(context);
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
  if (!indexElementType || indexElementType.type() != eltwise::DataType::INTX) {
    throw std::runtime_error("'gather' requires the data type for the second argument to be INTX.");
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
  auto elementType = ScalarType::get(tensor->getContext(), eltwise::DataType::INTX);
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
    SmallVector<Value*, 4> dims(prngOp.dims());
    auto newOp = rewriter.create<PrngOp>(op->getLoc(), resultType, prngOp.state(), dims);
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
    return Pattern::matchSuccess();
  }
};

void PrngOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<PrngCanonicalizer>(context);
}

Type PrngOp::getResultType(ArrayRef<Value*> operands) {
  IVLOG(5, "PrngOp::getResultType>")
  if (operands.size() < 2) {
    throw std::runtime_error("PrngOp requires at least 2 operands");
  }
  auto tensor = operands.front();
  auto dims = operands.drop_front();
  auto tensorType = eltwise::getRankedTensorType(tensor->getType());
  auto elementType = tensorType.getElementType();
  auto shape = eltwise::ComputeShape(dims);
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
  if (!indexElementType || indexElementType.type() != eltwise::DataType::INTX) {
    throw std::runtime_error("'scatter' requires the data type for the second argument to be INTX.");
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

#include "pmlc/dialect/tile/interfaces.cc.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/tile/ops.cc.inc"

}  // namespace pmlc::dialect::tile
