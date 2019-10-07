// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/ops.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/DebugStringHelper.h"

#include "base/util/logging.h"
#include "pmlc/dialect/eltwise/util.h"

namespace pmlc {
namespace dialect {
namespace tile {

using eltwise::constFoldBinaryOp;
using eltwise::constFoldUnaryOp;
using eltwise::m_One;
using eltwise::m_Zero;
using mlir::FloatAttr;
using mlir::IntegerAttr;
using mlir::OpRewritePattern;
using mlir::PatternMatchResult;
using mlir::PatternRewriter;

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

    eltwise::UpdateFuncOpType(newOp.getOperation());

    return matchSuccess();
  }
};

}  // namespace

void AffineDomainOp::getCanonicalizationPatterns(  //
    OwningRewritePatternList& results,             //
    MLIRContext* context) {
  results.insert<AffineDomainFolder>(context);
}

#include "pmlc/dialect/tile/ops_interfaces.cpp.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/tile/ops.cpp.inc"

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
