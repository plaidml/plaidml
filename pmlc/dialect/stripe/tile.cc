// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/tile.h"

namespace pmlc {
namespace dialect {
namespace stripe {

void Tile(ParallelForOp op, llvm::ArrayRef<int64_t> tile_sizes) {
  /*
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
  OpBuilder ibuild = inner.getBodyBuilder();

  auto aff_type = AffineType::get(ibuild.getContext());

  // Compute the new ranges, make the new affines, and add new constraints
  llvm::SmallVector<int64_t, 8> new_ranges;
  for (size_t i = 0; i < op.ranges().size(); i++) {
    // Compute new range and save it for later updating
    new_ranges.push_back((op.getRange(i) + tile_sizes[i] - 1) / tile_sizes[i]);
    // Recreate original index value
    auto mouter = ibuild.create<AffineMulOp>(op.getLoc(), ibody->getArgument(i), tile_sizes[i]);
    auto computed = ibuild.create<AffineAddOp>(op.getLoc(), mouter, ibody->getArgument(i));
    // new_values.push_back(computed);
    obody->getArgument(i)->replaceAllUsesWith(computed);
    mouter.getOperation()->setOperand(0, obody->getArgument(i));

    // Check if it's an uneven split, and if so, make a constraint
    if (op.getRange(i) % tile_sizes[i] != 0) {
      auto neg = ibuild.create<AffineMulOp>(op.getLoc(), computed, int64_t(-1));
      auto orange = ibuild.create<AffineConstOp>(op.getLoc(), aff_type, ibuild.getI64IntegerAttr(op.getRange(i) - 1));
      auto cexpr = ibuild.create<AffineAddOp>(op.getLoc(), neg, orange);
      auto aif = ibuild.create<ConstraintOp>(op.getLoc(), cexpr);
      ibody = new Block();
      aif.getOperation()->getRegion(0).push_back(ibody);
      ibuild.setInsertionPointToStart(ibody);
      ibuild.create<TerminateOp>(op.getLoc());
      ibuild.setInsertionPointToStart(ibody);
    }
  }
  // Update outer ranges
  op.getOperation()->setAttr("ranges", builder.getI64ArrayAttr(new_ranges));
  // Move the rest of the code into the interior
  ibody->getOperations().splice(std::prev(ibody->end(), 1), obody->getOperations(), obody->begin(),
                                std::prev(obody->end(), 2));
  */
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
