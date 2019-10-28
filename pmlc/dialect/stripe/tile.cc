// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/tile.h"

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

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
