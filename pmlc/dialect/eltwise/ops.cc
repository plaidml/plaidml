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
#include "pmlc/util/util.h"

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
    auto tensorType = getRankedTensorType(tensor->getType());
    auto resultTensorType = getRankedTensorType(castOp.result()->getType());
    auto elementType = resultTensorType.getElementType();
    auto resultType = RankedTensorType::get(tensorType.getShape(), elementType);
    if (resultType == castOp.result()->getType()) {
      return Pattern::matchFailure();
    }
    auto newOp = rewriter.create<OpType>(op->getLoc(), resultType, tensor);
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
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
  auto tensorType = getRankedTensorType(tensor->getType());
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
  auto tensorType = getRankedTensorType(tensor->getType());
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
  auto tensorType = getRankedTensorType(tensor->getType());
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
      util::UpdateFuncOpType(newOp.getOperation());
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

#include "pmlc/dialect/eltwise/interfaces.cc.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/eltwise/ops.cc.inc"

}  // namespace eltwise
}  // namespace dialect
}  // namespace pmlc
