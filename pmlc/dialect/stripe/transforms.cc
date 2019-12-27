// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/transforms.h"

#include <algorithm>
#include <thread>

#include "base/util/logging.h"

#include "pmlc/dialect/stripe/analysis.h"
#include "pmlc/dialect/stripe/util.h"

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

void ExtractConstraintCase(ParallelForOp pf, bool ge) {
  auto block = &pf.inner().front();
  auto con_it = std::prev(block->end(), 2);
  auto con = mlir::cast<ConstraintOp>(*con_it);
  auto region = ge ? &con.ge_case() : &con.lt_case();
  if (!region->empty()) {
    auto inner = &region->front();
    block->getOperations().splice(con_it, inner->getOperations(), inner->begin(), std::prev(inner->end(), 1));
  }
  con.erase();
}

void LimitLower(ParallelForOp op, size_t arg, int64_t val) {
  val = std::min(std::max(val, int64_t(0)), op.getRange(arg));
  auto body = &op.inner().front();
  OpBuilder builder(op);
  llvm::SmallVector<int64_t, 8> ranges;
  for (size_t i = 0; i < op.ranges().size(); i++) {
    ranges.push_back(i == arg ? op.getRange(i) - val : op.getRange(i));
  }
  op.getOperation()->setAttr("ranges", builder.getI64ArrayAttr(ranges));
  builder.setInsertionPointToStart(body);
  auto new_idx = builder.create<AffinePolyOp>(op.getLoc(), AffinePolynomial(val));
  body->getArgument(arg)->replaceAllUsesWith(new_idx);
  new_idx.setAttr("coeffs", builder.getI64ArrayAttr({1}));
  new_idx.getOperation()->setOperands({body->getArgument(arg)});
}

void LimitUpper(ParallelForOp op, size_t arg, int64_t val) {
  val = std::min(std::max(val, int64_t(0)), op.getRange(arg));
  OpBuilder builder(op);
  llvm::SmallVector<int64_t, 8> ranges;
  for (size_t i = 0; i < op.ranges().size(); i++) {
    ranges.push_back(i == arg ? val : op.getRange(i));
  }
  op.getOperation()->setAttr("ranges", builder.getI64ArrayAttr(ranges));
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

void LiftConstraint(ParallelForOp pf) {
  // Extract the constraint op
  auto pf_block = &pf.inner().front();
  auto con = mlir::cast<ConstraintOp>(*std::prev(pf_block->end(), 2));
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
  auto ge_pf = mlir::cast<ParallelForOp>(builder.clone(*pf.getOperation()));
  builder.create<TerminateOp>(loc);
  // Splice ge block out
  ExtractConstraintCase(ge_pf, true);
  builder.createBlock(&oc.lt_case(), oc.lt_case().begin());
  auto ic = builder.create<ConstraintOp>(loc, builder.create<AffinePolyOp>(loc, opoly + AffinePolynomial(irange.max)));
  builder.createBlock(&ic.ge_case(), ic.ge_case().begin());
  builder.clone(*pf.getOperation());
  builder.create<TerminateOp>(loc);
  if (!con.lt_case().empty()) {
    builder.createBlock(&ic.lt_case(), ic.lt_case().begin());
    auto lt_pf = mlir::cast<ParallelForOp>(builder.clone(*pf.getOperation()));
    builder.create<TerminateOp>(loc);
    ExtractConstraintCase(lt_pf, true);
  }
  builder.setInsertionPointToEnd(&oc.lt_case().front());
  builder.create<TerminateOp>(loc);
  pf.getOperation()->erase();
}

// Parallelize a ParallelForOip for eltwise only
void ParallelizeEltwise(ParallelForOp op, unsigned min_inner_size, const std::string& thread_tag) {
  unsigned n_processors = std::thread::hardware_concurrency();
  int64_t inner_size = 1;
  int64_t outer_size = 1;
  for (unsigned i = 0; i < op.ranges().size(); ++i) {
    outer_size *= op.getRange(i);
  }
  llvm::SmallVector<int64_t, 8> tiles(op.ranges().size());
  // Get the minimal inner tile
  for (int i = op.ranges().size() - 1; i >= 0; --i) {
    int64_t range = op.getRange(i);
    if (range * inner_size <= min_inner_size) {
      // The top priority is to satisfy the minimal inner size
      inner_size *= range;
      outer_size /= range;
      tiles[i] = range;
    }
    else {
      int64_t tile = range;
      for (int64_t k = range; k >= 1; --k) {
        if (range % k == 0 && k * inner_size >= min_inner_size) {
          // k is a feasible tile
          tile = k;
          if (outer_size / k >= n_processors) {
            // If outer_size / k is already enough for processor utilization,
            // we do not need smaller k.
            break;
          }
        }
      }
      tiles[i] = tile;
      inner_size *= tile;
      outer_size /= tile;
    }
  }
  // If outer_size is 1, we do not have to parallelize the loop
  if (outer_size > 1) {
    // Transform
    Tile(op, tiles);
    setOpAttrUnit(op, op.getBodyBuilder(), thread_tag);
  }
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
