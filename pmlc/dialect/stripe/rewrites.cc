// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/rewrites.h"

#include "mlir/IR/PatternMatch.h"
#include "pmlc/dialect/stripe/analysis.h"
#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/ops.h"
#include "pmlc/dialect/stripe/transforms.h"

namespace pmlc::dialect::stripe {

mlir::PatternMatchResult SimplifyPoly::matchAndRewrite(AffinePolyOp op, mlir::PatternRewriter& rewriter) const {
  bool has_non_canonical = false;
  for (size_t i = 0; i < op.getNumOperands(); i++) {
    if (!mlir::isa<mlir::BlockArgument>(op.getOperand(i))) {
      has_non_canonical = true;
    }
  }
  AffinePolynomial poly(op.result());
  if (poly.constant == 0 && poly.terms.size() == 1 && poly.terms.begin()->second == 1) {
    rewriter.replaceOp(op, poly.terms.begin()->first);
    return matchSuccess();
  }
  if (has_non_canonical) {
    rewriter.replaceOpWithNewOp<AffinePolyOp>(op, poly);
    return matchSuccess();
  }
  return matchFailure();
}

mlir::PatternMatchResult SimplifyNopRefines::match(RefineOp op) const {
  for (auto* offset : op.offsets()) {
    if (AffinePolynomial(offset) != AffinePolynomial()) {
      return matchFailure();
    }
  }
  if (op.getAttr(Dialect::getStripeAttrsName())) {
    return matchFailure();
  }
  return matchSuccess();
}

void SimplifyNopRefines::rewrite(RefineOp op, mlir::PatternRewriter& rewriter) const {
  rewriter.replaceOp(op, op.in());
}

mlir::PatternMatchResult InlineNoIndexParallelFors::match(ParallelForOp op) const {
  if (op.ranges().size() > 0) {
    return matchFailure();
  }
  if (op.getAttr(Dialect::getStripeAttrsName())) {
    return matchFailure();
  }
  return matchSuccess();
}

void InlineNoIndexParallelFors::rewrite(ParallelForOp op, mlir::PatternRewriter& rewriter) const {
  auto oblock = op.getOperation()->getBlock();
  auto iblock = &op.inner().front();
  oblock->getOperations().splice(Block::iterator(op), iblock->getOperations(), iblock->begin(),
                                 std::prev(iblock->end(), 1));
  rewriter.eraseOp(op);
}

mlir::PatternMatchResult RemoveRangeZeroParallelFors::match(ParallelForOp op) const {
  for (size_t i = 0; i < op.ranges().size(); i++) {
    if (op.getRange(i) == 0) {
      return matchSuccess();
    }
  }
  return matchFailure();
}

mlir::PatternMatchResult RemoveNoSideEffectParallelFors::match(ParallelForOp op) const {
  auto body = &op.inner().front();
  auto it_end = std::prev(body->end(), 1);
  for (auto it = body->begin(); it != it_end; ++it) {
    if (!it->hasNoSideEffect()) {
      return matchFailure();
    }
  }
  return matchSuccess();
}

mlir::PatternMatchResult RemoveRangeOneIndexes::match(ParallelForOp op) const {
  if (op.getAttr(Dialect::getStripeAttrsName())) {
    return matchFailure();
  }
  for (size_t i = 0; i < op.ranges().size(); i++) {
    if (op.getRange(i) == 1) {
      return matchSuccess();
    }
  }
  return matchFailure();
}

void RemoveRangeOneIndexes::rewrite(ParallelForOp op, mlir::PatternRewriter& rewriter) const {
  llvm::SmallVector<int64_t, 8> ranges;
  auto zero = rewriter.create<AffinePolyOp>(op.getLoc(), AffinePolynomial());
  auto body = &op.inner().front();
  auto idx_names = op.getAttrOfType<ArrayAttr>("idx_names");
  llvm::SmallVector<Attribute, 8> new_idx_names;
  for (size_t i = 0; i < op.ranges().size(); i++) {
    if (op.getRange(i) == 1) {
      body->getArgument(i)->replaceAllUsesWith(zero);
    } else {
      ranges.push_back(op.getRange(i));
      if (idx_names && idx_names.getValue().size() > i) {
        new_idx_names.push_back(idx_names.getValue()[i]);
      }
    }
  }
  auto rop = rewriter.create<ParallelForOp>(op.getLoc(), ranges);
  if (idx_names) {
    rop.setAttr("idx_names", rewriter.getArrayAttr(new_idx_names));
  }
  auto rbody = &rop.inner().front();
  size_t new_id = 0;
  for (size_t i = 0; i < op.ranges().size(); i++) {
    if (op.getRange(i) != 1) {
      body->getArgument(i)->replaceAllUsesWith(rbody->getArgument(new_id++));
    }
  }
  if (auto stripe_attr = op.getAttr(Dialect::getStripeAttrsName())) {
    rop.setAttr(Dialect::getStripeAttrsName(), stripe_attr);
  }
  rbody->getOperations().splice(std::prev(rbody->end(), 1), body->getOperations(), body->begin(),
                                std::prev(body->end(), 1));
  rewriter.eraseOp(op);
}

mlir::PatternMatchResult RemoveTrivialConstraints::matchAndRewrite(ConstraintOp op,
                                                                   mlir::PatternRewriter& rewriter) const {
  auto irange = AffineRange(AffinePolynomial(op.input()));
  auto oblock = op.getOperation()->getBlock();
  if (irange.min >= 0) {
    // Always true
    if (!op.ge_case().empty()) {
      auto iblock = &op.ge_case().front();
      oblock->getOperations().splice(Block::iterator(op), iblock->getOperations(), iblock->begin(),
                                     std::prev(iblock->end(), 1));
    }
    rewriter.eraseOp(op);
    return matchSuccess();
  } else if (irange.max < 0) {
    // Always false
    if (!op.lt_case().empty()) {
      auto iblock = &op.lt_case().front();
      oblock->getOperations().splice(Block::iterator(op), iblock->getOperations(), iblock->begin(),
                                     std::prev(iblock->end(), 1));
    }
    rewriter.eraseOp(op);
    return matchSuccess();
  } else {
    // Not trivial, never mind
    return matchFailure();
  }
}

mlir::PatternMatchResult SplitParallelFor::matchAndRewrite(ParallelForOp pf, mlir::PatternRewriter& rewriter) const {
  if (!SafeConstraintInterior(pf)) {
    return matchFailure();
  }
  // Extract the constraint op + poly
  auto pf_block = &pf.inner().front();
  auto con = mlir::dyn_cast<ConstraintOp>(*std::prev(pf_block->end(), 2));
  auto poly = AffinePolynomial(con.input());
  if (poly.terms.size() != 1) {
    return matchFailure();  // Only works for simple polynomials
  }
  // Get the actual relevant numbers from the poly
  mlir::BlockArgument* ba = poly.terms.begin()->first;
  // Find which argument it is (or not in this PF)
  int which_arg = -1;
  for (size_t i = 0; i < pf_block->getNumArguments(); i++) {
    if (pf_block->getArgument(i) == ba) {
      which_arg = i;
      break;
    }
  }
  if (which_arg < 0) {
    return matchFailure();
  }
  int64_t m = poly.terms.begin()->second;
  int64_t b = poly.constant;
  // Copy loop twice, once with each interior
  auto ge_pf = mlir::cast<ParallelForOp>(rewriter.clone(*pf.getOperation()));
  ExtractConstraintCase(ge_pf, true);
  auto lt_pf = mlir::cast<ParallelForOp>(rewriter.clone(*pf.getOperation()));
  ExtractConstraintCase(lt_pf, false);
  // Constraint says: m * i + b >= 0
  // We want to solve for an inequality in i:
  // If m > 0, i >= ceil(-b / m)
  // If m < 0, i <= floor(-b / m) or ! i >= floor(-b / m) + 1
  if (m > 0) {
    int64_t is = (-b + (m - 1)) / m;
    LimitLower(ge_pf, which_arg, is);
    LimitUpper(lt_pf, which_arg, is);
  } else {
    int64_t is = (-b / m) + 1;
    LimitLower(lt_pf, which_arg, is);
    LimitUpper(ge_pf, which_arg, is);
  }
  // Remove original
  rewriter.eraseOp(pf);
  return matchSuccess();
}

}  // namespace pmlc::dialect::stripe
