// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

namespace pmlc::dialect::pxa {

using mlir::AffineLoadOp;
using mlir::AffineParallelOp;
using mlir::Operation;

using WriteRead = std::pair<AffineReduceOp, AffineLoadOp>;
using WriteWrite = std::pair<AffineReduceOp, AffineReduceOp>;

/*
struct IndexMatcher {
  unsigned jointIndexes;
  std::map<BlockArgument, unsigned> jointA;
  std::map<BlockArgument, unsigned> jointB;
  bool unionIndexes(StrideInfo& a, StrideInfo& b) {

  }
};
*/

struct FusionPass : public FusionBase<FusionPass> {
  // Find the original source of a write
  AffineReduceOp findSourceWrite(Value val) {
    auto opRes = val.dyn_cast<mlir::OpResult>();
    if (auto op = mlir::dyn_cast_or_null<AffineReduceOp>(opRes.getOwner())) {
      return op;
    }
    if (auto op = mlir::dyn_cast<AffineParallelOp>(opRes.getOwner())) {
      auto retOp =
          mlir::cast<mlir::AffineYieldOp>(op.getBody()->getTerminator());
      return findSourceWrite(retOp.getOperand(opRes.getResultNumber()));
    }
    return AffineReduceOp();
  }

  // Attempts to fuse two ops if they look good.  We must fuse A into B, and
  // erase B to keep the outer loop happy
  void attemptFusion(AffineParallelOp a, AffineParallelOp b) {
    // First, we find all the write/read and write/write pairs, where block A
    // writes to a value that block B reads from or writes into.
    llvm::SmallVector<WriteRead, 4> writeReads;
    llvm::SmallVector<WriteWrite, 4> writeWrites;
    IVLOG(1, "Collectiong read/write information");
    // For each output from loop
    for (auto res : a.results()) {
      // Find the source write
      auto write = findSourceWrite(res);
      // If it's not a proper affine reduce, give up
      if (!write)
        return;
      // For each use of the write:
      for (auto user : res.getUsers()) {
        // Check if it is inside B, if not, we don't care, check next use.
        if (!b.getOperation()->isAncestor(user))
          continue;
        // Now we make sure it's a read or a write, if not, we can't do fusion,
        // bail.
        if (auto read = mlir::dyn_cast<AffineLoadOp>(user)) {
          writeReads.emplace_back(write, read);
        } else if (auto write2 = mlir::dyn_cast<AffineReduceOp>(user)) {
          writeWrites.emplace_back(write, write2);
        } else {
          return;
        }
      }
    }
    // Log how many write/reads and write/writes we found
    IVLOG(1, "Found " << writeReads.size() << " write/reads");
    IVLOG(1, "Found " << writeWrites.size() << " write/writes");
  }
  void runOnFunction() final {
    auto func = getFunction();
    // Autotile only the outermost loops
    auto &block = func.getBody().front();
    for (auto op_it = block.begin(); op_it != block.end();) {
      // See if the top op is an affine parallel
      auto fuseA = mlir::dyn_cast<AffineParallelOp>(*op_it);
      // Kick the iterator forward right away so if we end up fusing the op
      // down into a successor, we don't cause issues
      ++op_it;
      // If it's not a AF, continue
      if (!fuseA)
        continue;
      // Find the 'nearest reader' block:  Walk over each output, find any
      // blocks that ther output. pick the block closest to the writer.  This
      // block is legal to fuse into, since there are no intermediating
      // dependencies.
      Operation *nearestReader = nullptr;
      for (auto res : fuseA.results()) {
        for (auto user : res.getUsers()) {
          Operation *op = block.findAncestorOpInBlock(*user);
          if (nearestReader == nullptr || op->isBeforeInBlock(nearestReader)) {
            nearestReader = op;
          }
        }
      }
      // Check if the closest reader is also and affine parallel, in which case,
      // attempt to merge
      if (auto fuseB =
              mlir::dyn_cast_or_null<AffineParallelOp>(nearestReader)) {
        attemptFusion(fuseA, fuseB);
      }
    }
  }
};

std::unique_ptr<mlir::Pass> createFusionPass() {
  return std::make_unique<FusionPass>();
}

} // namespace pmlc::dialect::pxa
