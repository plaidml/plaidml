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

namespace pmlc {
namespace dialect {
namespace tile {

using eltwise::constFoldBinaryOp;
using eltwise::constFoldUnaryOp;
using eltwise::m_One;
using eltwise::m_Zero;
using llvm::SmallVector;
using mlir::FloatAttr;
using mlir::IntegerAttr;
using mlir::OpRewritePattern;
using mlir::PatternMatchResult;
using mlir::PatternRewriter;
using mlir::Value;

OpFoldResult AffineConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

OpFoldResult AffineAddOp::fold(ArrayRef<Attribute> operands) {
  /// add(x, 0) -> x
  if (matchPattern(rhs(), m_Zero())) {
    return lhs();
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a + b; });
}

OpFoldResult AffineDivOp::fold(ArrayRef<Attribute> operands) {
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
    return Builder(getContext()).getZeroAttr(getType());
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a / b; });
}

OpFoldResult AffineMulOp::fold(ArrayRef<Attribute> operands) {
  // mul(x, 0) -> 0
  if (matchPattern(rhs(), m_Zero())) {
    return rhs();
  }
  // mul(x, 1) -> x
  if (matchPattern(rhs(), m_One())) {
    return lhs();
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a * b; });
}

OpFoldResult AffineNegOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp(operands, [](double x) { return -x; });
}

OpFoldResult AffineSubOp::fold(ArrayRef<Attribute> operands) {
  // sub(x, x) -> 0
  if (lhs() == rhs()) {
    return Builder(getContext()).getZeroAttr(getType());
  }
  /// sub(x, 0) -> x
  if (matchPattern(rhs(), m_Zero())) {
    return lhs();
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a - b; });
}

OpFoldResult DimOp::fold(ArrayRef<Attribute> operands) {
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
    auto terminator = &op.body().front().back();
    auto size_map_op = llvm::dyn_cast<AffineSizeMapOp>(terminator->getOperand(0)->getDefiningOp());
    if (!size_map_op) {
      return matchFailure();
    }
    llvm::SmallVector<Value*, 4> sizes(size_map_op.sizes());
    auto shape = eltwise::ComputeShape(sizes);
    auto existingType = op.getType().cast<RankedTensorType>();
    auto tensorType = rewriter.getTensorType(shape, existingType.getElementType());
    IVLOG(6, "  existingType: " << mlir::debugString(existingType));
    IVLOG(6, "  tensorType: " << mlir::debugString(tensorType));
    if (existingType == tensorType) {
      return matchFailure();
    }
    auto newOp = rewriter.create<AffineDomainOp>(op.getLoc(), tensorType);
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
  if (!indexElementType || indexElementType.type() != eltwise::DataType::INT32) {
    throw std::runtime_error("'gather' requires the data type for the second argument to be INT32.");
  }
  // std::vector<std::shared_ptr<DimExpr>> dims;
  // for (size_t i = 0; i < index->shape.dims.size(); i++) {
  //   dims.push_back(index->shape.dims[i].expr);
  // }
  // for (size_t i = 1; i < data->shape.dims.size(); i++) {
  //   dims.push_back(data->shape.dims[i].expr);
  // }
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
    auto newOp = rewriter.create<IndexOp>(op->getLoc(), resultType, indexOp.tensor(), indexOp.dim());
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
  if (operands.size() != 2) {
    throw std::runtime_error("IndexOp requires 2 operands");
  }
  auto tensor = operands.front();
  auto tensorType = eltwise::getRankedTensorType(tensor->getType());
  // auto elementType = IndexType::get(tensor->getContext());
  auto elementType = ScalarType::get(tensor->getContext(), eltwise::DataType::INT32);  // TODO: index type?
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
  auto elementType = tensorType.getElementType();
  return RankedTensorType::get({tensorType.getRank()}, elementType);
}

#include "pmlc/dialect/tile/ops_interfaces.cpp.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/tile/ops.cpp.inc"

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
