// Copyright 2019, Intel Corporation

#include "pmlc/dialect/eltwise/ops.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "llvm/ADT/StringSwitch.h"

#include "mlir/Dialect/StandardOps/Ops.h"
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

using mlir::FloatAttr;
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

struct CastCanonicalizer : public OpRewritePattern<CastOp> {
  using OpRewritePattern<CastOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(CastOp castOp, PatternRewriter& rewriter) const override {
    IVLOG(5, "CastCanonicalizer::matchAndRewrite> " << mlir::debugString(castOp));
    auto op = castOp.getOperation();
    auto tensor = castOp.tensor();
    auto tensorType = getRankedTensorType(tensor->getType());
    auto existingType = getRankedTensorType(castOp.result()->getType());
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

Type CastOp::getResultType(ArrayRef<Value*> operands) {  //
  llvm_unreachable("CastOp::getResultType not implemented");
}

Operation* CastOp::create(OpBuilder* builder, Location loc, Type type, ArrayRef<Value*> operands) {
  OperationState state(loc, getOperationName());
  state.addOperands(operands);
  state.addTypes(type);
  return builder->createOperation(state);
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

Type SelectOp::getResultType(ArrayRef<Value*> operands) {
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

#include "pmlc/dialect/eltwise/interfaces.cc.inc"

#define fi_to_std(orig, stdf, stdi)                                                                                 \
  Value* orig::buildStandard(OpBuilder& builder, mlir::Location loc, ScalarType stype, ArrayRef<Value*> operands) { \
    if (is_float(stype.type())) {                                                                                   \
      return builder.create<mlir::stdf>(loc, operands[0], operands[1]);                                             \
    } else {                                                                                                        \
      return builder.create<mlir::stdi>(loc, operands[0], operands[1]);                                             \
    }                                                                                                               \
  }

fi_to_std(AddOp, AddFOp, AddIOp) fi_to_std(SubOp, SubFOp, SubIOp) fi_to_std(MulOp, MulFOp, MulIOp)
#define fus_to_std(orig, stdf, stdu, stds)                                                                          \
  Value* orig::buildStandard(OpBuilder& builder, mlir::Location loc, ScalarType stype, ArrayRef<Value*> operands) { \
    if (is_float(stype.type())) {                                                                                   \
      return builder.create<mlir::stdf>(loc, operands[0], operands[1]);                                             \
    } else if (is_uint(stype.type())) {                                                                             \
      return builder.create<mlir::stdu>(loc, operands[0], operands[1]);                                             \
    } else {                                                                                                        \
      return builder.create<mlir::stds>(loc, operands[0], operands[1]);                                             \
    }                                                                                                               \
  }

    fus_to_std(DivOp, DivFOp, DivIUOp, DivISOp) fus_to_std(ModOp, RemFOp, RemIUOp, RemISOp)

        Value* MinOp::buildStandard(OpBuilder& builder, mlir::Location loc, ScalarType stype,
                                    ArrayRef<Value*> operands) {
  return nullptr;
}
Value* MaxOp::buildStandard(OpBuilder& builder, mlir::Location loc, ScalarType stype, ArrayRef<Value*> operands) {
  return nullptr;
}
Value* AndOp::buildStandard(OpBuilder& builder, mlir::Location loc, ScalarType stype, ArrayRef<Value*> operands) {
  return nullptr;
}
Value* OrOp::buildStandard(OpBuilder& builder, mlir::Location loc, ScalarType stype, ArrayRef<Value*> operands) {
  return nullptr;
}
Value* XorOp::buildStandard(OpBuilder& builder, mlir::Location loc, ScalarType stype, ArrayRef<Value*> operands) {
  return nullptr;
}
Value* ShlOp::buildStandard(OpBuilder& builder, mlir::Location loc, ScalarType stype, ArrayRef<Value*> operands) {
  return nullptr;
}
Value* ShrOp::buildStandard(OpBuilder& builder, mlir::Location loc, ScalarType stype, ArrayRef<Value*> operands) {
  return nullptr;
}
Value* PowOp::buildStandard(OpBuilder& builder, mlir::Location loc, ScalarType stype, ArrayRef<Value*> operands) {
  return nullptr;
}

#define GET_OP_CLASSES
#include "pmlc/dialect/eltwise/ops.cc.inc"

}  // namespace eltwise
}  // namespace dialect
}  // namespace pmlc
