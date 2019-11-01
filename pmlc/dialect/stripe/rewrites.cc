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
    if (op.getOperand(i)->getKind() != Value::Kind::BlockArgument) {
      has_non_canonical = true;
    }
  }
  AffinePolynomial a(op.result());
  if (a.constant == 0 && a.terms.size() == 1 && a.terms.begin()->second == 1) {
    rewriter.replaceOp(op, a.terms.begin()->first);
    return matchSuccess();
  }
  if (has_non_canonical) {
    rewriter.replaceOpWithNewOp<AffinePolyOp>(op, a);
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

void SimplifyNopRefines::rewrite(RefineOp op, mlir::PatternRewriter& rewriter) const {  // NOLINT(runtime/references)
  rewriter.replaceOp(op, op.in());
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
    rewriter.replaceOp(op, llvm::None);
    return matchSuccess();
  } else if (irange.max < 0) {
    // Always false
    if (!op.lt_case().empty()) {
      auto iblock = &op.lt_case().front();
      oblock->getOperations().splice(Block::iterator(op), iblock->getOperations(), iblock->begin(),
                                     std::prev(iblock->end(), 1));
    }
    rewriter.replaceOp(op, llvm::None);
    return matchSuccess();
  } else {
    // Not trivial, never mind
    return matchFailure();
  }
}

// 'Lifts' a constraint.  The idea here is more move 'most' of the
//  checks out of a parallel for, and to split the cases up to allow
//  more constraint rewrites, eventually resulting in (hopefully) no
//  constraints in the fast path.
//
// More exactly, translates:
//
// PF [inner] {
//   Pre
//   Con ([outer_poly] + [inner_poly] + const) >= 0 {
//     GE Case
//   } else {
//     LT Case
//   }
// }
//
// into:
//
// Con ([outer_poly] + min(inner_poly) + const) >= 0 {
//   PF [inner] {
//     Pre
//     GE Case
//   }
// } else {
//   Cons ([outer_poly] + max(inner_poly) + const >= 0) {
//     PF [inner] {
//       Pre
//       Con ([outer_poly] + [inner_poly] + const) >= 0 {
//         GE Case
//       } else {
//         LT Case
//       }
//     }
//   } else {
//     PF [inner] {
//       Pre
//       LT Case
//     }
//   }
// }
//

mlir::PatternMatchResult LiftConstraints::match(ParallelForOp op) const {
  if (SafeConstraintInterior(op)) {
    return matchSuccess();
  } else {
    return matchFailure();
  }
}

void LiftConstraints::rewrite(ParallelForOp pf, mlir::PatternRewriter& rewriter) const {
  // Extract the constraint op
  auto pf_block = &pf.inner().front();
  auto con = mlir::dyn_cast<ConstraintOp>(*std::prev(pf_block->end(), 2));
  // Get the complete polynomial, and split it into innner + outer
  auto opoly = AffinePolynomial(con.input());
  AffinePolynomial ipoly;
  for (mlir::BlockArgument* arg : pf_block->getArguments()) {
    auto it = opoly.terms.find(arg);
    if (it != opoly.terms.end()) {
      ipoly.terms.emplace(*it);
      opoly.terms.erase(it);
    }
  }
  // Compute the range of the inner polynomial
  auto irange = AffineRange(ipoly);
  // Make a builder aimed right before the original parallel-for
  OpBuilder builder(pf.getOperation());
  Location loc = pf.getLoc();
  // Build the rewritten version
  auto oc = builder.create<ConstraintOp>(loc, builder.create<AffinePolyOp>(loc, opoly + AffinePolynomial(irange.min)));
  // Make always true block
  builder.createBlock(&oc.ge_case(), oc.ge_case().begin());
  auto ge_pf = mlir::dyn_cast<ParallelForOp>(builder.clone(*pf.getOperation()));
  builder.create<TerminateOp>(loc);
  // Splice ge block out
  auto ge_block = &ge_pf.inner().front();
  auto ge_con_it = std::prev(ge_block->end(), 2);
  auto ge_con = mlir::dyn_cast<ConstraintOp>(*ge_con_it);
  auto ge_ge = &ge_con.ge_case().front();
  ge_block->getOperations().splice(ge_con_it, ge_ge->getOperations(), ge_ge->begin(), std::prev(ge_ge->end(), 1));
  ge_con.erase();
  builder.createBlock(&oc.lt_case(), oc.lt_case().begin());
  auto ic = builder.create<ConstraintOp>(loc, builder.create<AffinePolyOp>(loc, opoly + AffinePolynomial(irange.max)));
  builder.createBlock(&ic.ge_case(), ic.ge_case().begin());
  builder.clone(*pf.getOperation());
  builder.create<TerminateOp>(loc);
  if (!con.lt_case().empty()) {
    builder.createBlock(&ic.lt_case(), ic.lt_case().begin());
    auto lt_pf = mlir::dyn_cast<ParallelForOp>(builder.clone(*pf.getOperation()));
    builder.create<TerminateOp>(loc);
    // Splice lt block out
    auto lt_block = &lt_pf.inner().front();
    auto lt_con_it = std::prev(lt_block->end(), 2);
    auto lt_con = mlir::dyn_cast<ConstraintOp>(*lt_con_it);
    auto lt_lt = &lt_con.lt_case().front();
    lt_block->getOperations().splice(lt_con_it, lt_lt->getOperations(), lt_lt->begin(), std::prev(lt_lt->end(), 1));
    lt_con.erase();
  }
  builder.setInsertionPointToEnd(&oc.lt_case().front());
  builder.create<TerminateOp>(loc);
  rewriter.replaceOp(pf, llvm::None);
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
  // mlir::BlockArgument* ba = poly.terms.begin()->first;
  // int64_t m = poly.terms.begin()->second;
  // int64_t b = poly.constant;
  // Constraint says: m * i + b >= 0
  // We want to solve for an inequality in i:
  // If m > 0, i >= ceil(-b / m)
  // If m < 0, i <= floor(-b / m)A
  // Copy loop twice, once with each interior
  auto ge_pf = mlir::cast<ParallelForOp>(rewriter.clone(*pf.getOperation()));
  ExtractConstraintCase(ge_pf, true);
  auto lt_pf = mlir::cast<ParallelForOp>(rewriter.clone(*pf.getOperation()));
  ExtractConstraintCase(lt_pf, false);
  /*
  if (m > 0) {
    int64_t is = (-b + (m - 1)) / m;
  */
  // Remove original
  rewriter.replaceOp(pf, llvm::None);
  return matchSuccess();
}

}  // namespace pmlc::dialect::stripe
