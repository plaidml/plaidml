// Copyright 2019, Intel Corporation

#include "pmlc/dialect/eltwise/ops.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "llvm/ADT/StringSwitch.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/DebugStringHelper.h"

#include "base/util/logging.h"
#include "pmlc/dialect/eltwise/util.h"

#define DEBUG_TYPE "eltwise"

namespace pmlc {
namespace dialect {
namespace eltwise {

using mlir::IntegerAttr;
using mlir::OpRewritePattern;
using mlir::Pattern;
using mlir::PatternMatchResult;
using mlir::PatternRewriter;

mlir::OpFoldResult ScalarConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

//
// ---- CastOp ----
//

template <typename OpType>
struct CastCanonicalizer : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(OpType castOp, PatternRewriter& rewriter) const override {
    IVLOG(5, "CastCanonicalizer::matchAndRewrite> " << mlir::debugString(castOp));
    auto op = castOp.getOperation();
    auto tensor = castOp.tensor();
    auto tensorType = GetTensorType(tensor->getType());
    auto resultTensorType = GetTensorType(castOp.result()->getType());
    auto elementType = resultTensorType.getElementType();
    auto resultType = RankedTensorType::get(tensorType.getShape(), elementType);
    if (resultType == castOp.result()->getType()) {
      return Pattern::matchFailure();
    }
    auto newOp = rewriter.create<OpType>(op->getLoc(), resultType, tensor);
    rewriter.replaceOp(op, {newOp});
    UpdateFuncOpType(newOp.getOperation());
    return Pattern::matchSuccess();
  }
};

void AsFloatOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<CastCanonicalizer<AsFloatOp>>(context);
}

Type AsFloatOp::getResultType(ArrayRef<Value*> operands) {
  IVLOG(5, "AsFloatOp::getResultType>")
  if (operands.size() != 2) {
    throw std::runtime_error("AsFloatOp requires 2 operands");
  }
  auto tensor = operands[0];
  auto bitwidthOp = operands[1]->getDefiningOp();
  IntegerAttr bitwidth;
  if (!m_Constant(&bitwidth).match(bitwidthOp)) {
    throw std::runtime_error("AsFloatOp requires 2nd operand to be a constant integer");
  }
  auto tensorType = GetTensorType(tensor->getType());
  ScalarType elementType;
  switch (bitwidth.getInt()) {
    case 16:
      elementType = ScalarType::get(tensor->getContext(), DataType::FLOAT16);
      break;
    case 32:
      elementType = ScalarType::get(tensor->getContext(), DataType::FLOAT32);
      break;
    case 64:
      elementType = ScalarType::get(tensor->getContext(), DataType::FLOAT64);
      break;
  }
  return RankedTensorType::get(tensorType.getShape(), elementType);
}

void AsIntOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<CastCanonicalizer<AsIntOp>>(context);
}

Type AsIntOp::getResultType(ArrayRef<Value*> operands) {
  IVLOG(5, "AsIntOp::getResultType>")
  if (operands.size() != 2) {
    throw std::runtime_error("AsIntOp requires 2 operands");
  }
  auto tensor = operands[0];
  auto bitwidthOp = operands[1]->getDefiningOp();
  IntegerAttr bitwidth;
  if (!m_Constant(&bitwidth).match(bitwidthOp)) {
    throw std::runtime_error("AsIntOp requires 2nd operand to be a constant integer");
  }
  auto tensorType = GetTensorType(tensor->getType());
  ScalarType elementType;
  switch (bitwidth.getInt()) {
    case 8:
      elementType = ScalarType::get(tensor->getContext(), DataType::INT8);
      break;
    case 16:
      elementType = ScalarType::get(tensor->getContext(), DataType::INT16);
      break;
    case 32:
      elementType = ScalarType::get(tensor->getContext(), DataType::INT32);
      break;
    case 64:
      elementType = ScalarType::get(tensor->getContext(), DataType::INT64);
      break;
  }
  return RankedTensorType::get(tensorType.getShape(), elementType);
}

void AsUIntOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<CastCanonicalizer<AsUIntOp>>(context);
}

Type AsUIntOp::getResultType(ArrayRef<Value*> operands) {
  IVLOG(5, "AsUIntOp::getResultType>")
  if (operands.size() != 2) {
    throw std::runtime_error("AsUIntOp requires 2 operands");
  }
  auto tensor = operands[0];
  auto bitwidthOp = operands[1]->getDefiningOp();
  IntegerAttr bitwidth;
  if (!m_Constant(&bitwidth).match(bitwidthOp)) {
    throw std::runtime_error("AsUIntOp requires 2nd operand to be a constant integer");
  }
  auto tensorType = GetTensorType(tensor->getType());
  ScalarType elementType;
  switch (bitwidth.getInt()) {
    case 8:
      elementType = ScalarType::get(tensor->getContext(), DataType::UINT8);
      break;
    case 16:
      elementType = ScalarType::get(tensor->getContext(), DataType::UINT16);
      break;
    case 32:
      elementType = ScalarType::get(tensor->getContext(), DataType::UINT32);
      break;
    case 64:
      elementType = ScalarType::get(tensor->getContext(), DataType::UINT64);
      break;
  }
  return RankedTensorType::get(tensorType.getShape(), elementType);
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
    UpdateFuncOpType(newOp.getOperation());
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
  auto tensorType = GetTensorType(tensor->getType());
  auto tensorElementType = tensorType.getElementType();
  if (!tensorType.getRank()) {
    throw std::runtime_error("'gather' requires first operand to have at least one dimension.");
  }
  auto index = operands[1];
  auto indexType = GetTensorType(index->getType());
  auto indexElementType = indexType.getElementType().dyn_cast<ScalarType>();
  if (!indexElementType || indexElementType.type() != DataType::INT32) {
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
    UpdateFuncOpType(newOp.getOperation());
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
  auto tensorType = GetTensorType(tensor->getType());
  // auto elementType = IndexType::get(tensor->getContext());
  auto elementType = ScalarType::get(tensor->getContext(), DataType::INT32);  // TODO: index type?
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
    UpdateFuncOpType(newOp.getOperation());
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
  auto tensorType = GetTensorType(tensor->getType());
  auto elementType = tensorType.getElementType();
  auto shape = ComputeShape(dims);
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
    UpdateFuncOpType(newOp.getOperation());
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
  auto tensorType = GetTensorType(tensor->getType());
  auto elementType = tensorType.getElementType();
  auto shape = ComputeShape(dims);
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
    UpdateFuncOpType(newOp.getOperation());
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
  auto tensorType = GetTensorType(tensor->getType());
  auto elementType = tensorType.getElementType();
  return RankedTensorType::get({tensorType.getRank()}, elementType);
}

//
// ---- EltwiseOp ----
//

template <typename OpType>
struct EltwiseCanonicalizer : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(OpType eltwiseOp, PatternRewriter& rewriter) const override {
    IVLOG(5, "EltwiseCanonicalizer::matchAndRewrite> " << mlir::debugString(eltwiseOp));
    auto op = eltwiseOp.getOperation();
    SmallVector<Value*, 2> operands(op->getOperands());
    auto resultType = OpType::getResultType(operands);
    if (resultType == eltwiseOp.result()->getType()) {
      return Pattern::matchFailure();
    }
    if (auto type = eltwiseOp.type().template dyn_cast<ScalarType>()) {
      auto newOp = rewriter.create<OpType>(op->getLoc(), type, operands);
      rewriter.replaceOp(op, {newOp});
      UpdateFuncOpType(newOp.getOperation());
      return Pattern::matchSuccess();
    }
    return Pattern::matchFailure();
  }
};

#define DEFINE_CANONICALIZER(_op_)                                                                  \
  void _op_::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) { \
    results.insert<EltwiseCanonicalizer<_op_>>(context);                                            \
  }

DEFINE_CANONICALIZER(AbsOp);
DEFINE_CANONICALIZER(ACosOp);
DEFINE_CANONICALIZER(AddOp);
DEFINE_CANONICALIZER(AndOp);
DEFINE_CANONICALIZER(ASinOp);
DEFINE_CANONICALIZER(AssignOp);
DEFINE_CANONICALIZER(ATanOp);
DEFINE_CANONICALIZER(CeilOp);
DEFINE_CANONICALIZER(CmpEqOp);
DEFINE_CANONICALIZER(CmpGeOp);
DEFINE_CANONICALIZER(CmpGtOp);
DEFINE_CANONICALIZER(CmpLeOp);
DEFINE_CANONICALIZER(CmpLtOp);
DEFINE_CANONICALIZER(CmpNeOp);
DEFINE_CANONICALIZER(CosHOp);
DEFINE_CANONICALIZER(CosOp);
DEFINE_CANONICALIZER(DivOp);
DEFINE_CANONICALIZER(ExpOp);
DEFINE_CANONICALIZER(FloorOp);
DEFINE_CANONICALIZER(IdentOp);
DEFINE_CANONICALIZER(LogOp);
DEFINE_CANONICALIZER(MaxOp);
DEFINE_CANONICALIZER(MinOp);
DEFINE_CANONICALIZER(ModOp);
DEFINE_CANONICALIZER(MulOp);
DEFINE_CANONICALIZER(NegOp);
DEFINE_CANONICALIZER(NotOp);
DEFINE_CANONICALIZER(OrOp);
DEFINE_CANONICALIZER(PowOp);
DEFINE_CANONICALIZER(ReluOp);
DEFINE_CANONICALIZER(RoundOp);
DEFINE_CANONICALIZER(ShlOp);
DEFINE_CANONICALIZER(ShrOp);
DEFINE_CANONICALIZER(SignOp);
DEFINE_CANONICALIZER(SinHOp);
DEFINE_CANONICALIZER(SinOp);
DEFINE_CANONICALIZER(SqrtOp);
DEFINE_CANONICALIZER(SubOp);
DEFINE_CANONICALIZER(TanHOp);
DEFINE_CANONICALIZER(TanOp);
DEFINE_CANONICALIZER(XorOp);
DEFINE_CANONICALIZER(SelectOp);

OpFoldResult MulOp::fold(ArrayRef<Attribute> operands) {
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

#include "pmlc/dialect/eltwise/ops_interfaces.cpp.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/eltwise/ops.cpp.inc"

}  // namespace eltwise
}  // namespace dialect
}  // namespace pmlc

#include "pmlc/dialect/eltwise/ops_enums.cpp.inc"
