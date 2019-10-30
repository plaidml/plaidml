// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/transforms.h"

#include "pmlc/dialect/stripe/analysis.h"

namespace pmlc {
namespace dialect {
namespace stripe {

void Tile(ParallelForOp op, llvm::ArrayRef<int64_t> tile_sizes) {
  // Make a builder to write to just before the terminator
  Block* obody = op.getBody();
  auto builder = op.getBodyBuilder();

  // Make the inner parallel for
  auto inner = builder.create<ParallelForOp>(op.getLoc(), tile_sizes);
  // Copy index names
  if (auto names = op.getOperation()->getAttr("idx_names")) {
    inner.getOperation()->setAttr("idx_names", names);
  }
  Block* ibody = inner.getBody();
  Block* cbody = ibody;
  OpBuilder cbuild = inner.getBodyBuilder();

  // Compute the new ranges, make the new affines, and add new constraints
  llvm::SmallVector<int64_t, 8> new_ranges;
  for (size_t i = 0; i < op.ranges().size(); i++) {
    // Compute new range and save it for later updating
    new_ranges.push_back((op.getRange(i) + tile_sizes[i] - 1) / tile_sizes[i]);

    // Make a polynomial to represent the computed index value, replace uses of
    // old value, then update poly to be the correct computation
    auto computed_idx = cbuild.create<AffinePolyOp>(op.getLoc(), AffinePolynomial());
    obody->getArgument(i)->replaceAllUsesWith(computed_idx);
    computed_idx.setAttr("coeffs", cbuild.getI64ArrayAttr({tile_sizes[i], 1}));
    computed_idx.getOperation()->setOperands({obody->getArgument(i), ibody->getArgument(i)});

    // Check if it's an uneven split, and if so, make a constraint
    if (op.getRange(i) % tile_sizes[i] != 0) {
      auto cpoly = AffinePolynomial(computed_idx.result()) * -1 + AffinePolynomial(op.getRange(i) - 1);
      auto cexpr = cbuild.create<AffinePolyOp>(op.getLoc(), cpoly);
      auto aif = cbuild.create<ConstraintOp>(op.getLoc(), cexpr);
      cbody = new Block();
      aif.getOperation()->getRegion(0).push_back(cbody);
      cbuild.setInsertionPointToStart(cbody);
      cbuild.create<TerminateOp>(op.getLoc());
      cbuild.setInsertionPointToStart(cbody);
    }
  }
  // Update outer ranges
  op.getOperation()->setAttr("ranges", builder.getI64ArrayAttr(new_ranges));
  // Move the rest of the code into the interior
  cbody->getOperations().splice(std::prev(cbody->end(), 1), obody->getOperations(), obody->begin(),
                                std::prev(obody->end(), 2));
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

LogicalResult LiftConstraint(ParallelForOp pf) {
  // Assert that final op is a safe constraint
  assert(SafeConstraintInterior(pf));
  // Extract that opA
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
  pf.getOperation()->erase();
  return mlir::success();
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
