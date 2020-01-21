// Copyright 2019, Intel Corporation

#include "pmlc/dialect/eltwise/ir/ops.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/eltwise/ir/util.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

#define DEBUG_TYPE "eltwise"

namespace pmlc::dialect::eltwise {

using mlir::FloatAttr;
using mlir::IntegerAttr;
using mlir::Location;
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

struct CastCanonicalizer : public OpRewritePattern<CastOp> {
  using OpRewritePattern<CastOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(CastOp castOp, PatternRewriter& rewriter) const override {
    IVLOG(5, "CastCanonicalizer::matchAndRewrite> " << mlir::debugString(castOp));
    auto op = castOp.getOperation();
    auto tensor = castOp.tensor();
    auto tensorType = getRankedTensorType(tensor.getType());
    auto existingType = getRankedTensorType(castOp.result().getType());
    auto elementType = existingType.getElementType();
    auto resultType = RankedTensorType::get(tensorType.getShape(), elementType);
    if (resultType == existingType) {
      return Pattern::matchFailure();
    }
    auto newOp = rewriter.create<CastOp>(op->getLoc(), resultType, tensor);
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
    return Pattern::matchSuccess();
  }
};

void CastOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<CastCanonicalizer>(context);
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
    auto operands = llvm::to_vector<2>(op->getOperands());
    auto resultType = OpType::getResultType(operands);
    if (resultType == eltwiseOp.result().getType()) {
      return Pattern::matchFailure();
    }
    auto newOp = rewriter.create<OpType>(op->getLoc(), operands);
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
    return Pattern::matchSuccess();
  }
};

#define DEFINE_CANONICALIZER(_op_)                                                                  \
  void _op_::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) { \
    results.insert<EltwiseCanonicalizer<_op_>>(context);                                            \
  }

DEFINE_CANONICALIZER(AbsOp);
DEFINE_CANONICALIZER(ACosOp);
DEFINE_CANONICALIZER(AddOp);
DEFINE_CANONICALIZER(ASinOp);
DEFINE_CANONICALIZER(AssignOp);
DEFINE_CANONICALIZER(ATanOp);
DEFINE_CANONICALIZER(BitAndOp);
DEFINE_CANONICALIZER(BitNotOp);
DEFINE_CANONICALIZER(BitOrOp);
DEFINE_CANONICALIZER(BitXorOp);
DEFINE_CANONICALIZER(BitShlOp);
DEFINE_CANONICALIZER(BitShrOp);
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
DEFINE_CANONICALIZER(PowOp);
DEFINE_CANONICALIZER(ReluOp);
DEFINE_CANONICALIZER(RoundOp);
DEFINE_CANONICALIZER(SignOp);
DEFINE_CANONICALIZER(SinHOp);
DEFINE_CANONICALIZER(SinOp);
DEFINE_CANONICALIZER(SqrtOp);
DEFINE_CANONICALIZER(SubOp);
DEFINE_CANONICALIZER(TanHOp);
DEFINE_CANONICALIZER(TanOp);
DEFINE_CANONICALIZER(SelectOp);

OpFoldResult AddOp::fold(ArrayRef<Attribute> operands) {
  /// add(x, 0) -> x
  if (matchPattern(rhs(), m_Zero())) {
    return lhs();
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a + b; });
}

OpFoldResult DivOp::fold(ArrayRef<Attribute> operands) {
  // don't fold division by zero
  // modeling this choice on DivUIOp::Fold from the standard dialect
  if (matchPattern(rhs(), m_Zero())) {
    return {};
  }
  // div(x, 1) -> x
  if (matchPattern(rhs(), m_One())) {
    return lhs();
  }
  // div(0, x) -> 0
  if (matchPattern(lhs(), m_Zero())) {
    return lhs();
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a / b; });
}

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

Type SelectOp::getResultType(ValueRange operands) {
  auto inferShapeType = getRankedTensorType(ComputeResultType(operands));
  auto inferElementType = getRankedTensorType(ComputeResultType(operands.drop_front()));
  return RankedTensorType::get(inferShapeType.getShape(), inferElementType.getElementType());
}

OpFoldResult SubOp::fold(ArrayRef<Attribute> operands) {
  // sub(x, 0) -> x
  if (matchPattern(rhs(), m_Zero())) {
    return lhs();
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a - b; });
}

#include "pmlc/dialect/eltwise/ir/interfaces.cc.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/eltwise/ir/ops.cc.inc"

}  // namespace pmlc::dialect::eltwise
