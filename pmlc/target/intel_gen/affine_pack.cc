// Copyright 2020 Intel Corporation

#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "pmlc/target/intel_gen/pass_detail.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/math/util.h"

#include "mlir/Support/DebugStringHelper.h"

using namespace mlir; // NOLINT

namespace pmlc::target::intel_gen {

namespace {
using llvm::DenseMap;
using llvm::DenseSet;
using llvm::SmallVector;
using pmlc::util::math::IsPo2;

struct IndexPacking {
  unsigned sourceIdx;
  uint64_t floorDiv;
  uint64_t mod;
};

// Pack N indexes into 3 indexes via floor div + mod, pack po2 indexes first to
// make floor div and mod more likely to be lowerable as bit operations
struct AffinePackPass : public AffinePackBase<AffinePackPass> {
  LogicalResult maybePack(AffineParallelOp op) {
    auto hardware = op.getAttrOfType<StringAttr>("hardware");
    if (!hardware || op.getIVs().size() <= 3) {
      // Doesn't need special handling
      return success();
    }
    // Check for unsupported cases
    if (op.getNumResults() != 0) {
      op.emitError("Unable to reduce dims to 3 for affine.parallel that "
                   "produces results");
      return failure();
    }
    auto maybeRanges = op.getConstantRanges();
    if (!maybeRanges) {
      op.emitError("Unable to reduce dims to 3 for non-constant ranges");
      return failure();
    }
    auto ranges = *maybeRanges;
    // Verify outer loop is mormalized and no trip-count 1 ranges
    for (unsigned i = 0; i < ranges.size(); i++) {
      auto lbExpr =
          op.lowerBoundsMap().getResult(i).dyn_cast<AffineConstantExpr>();
      if (!lbExpr || lbExpr.getValue() != 0 || op.getSteps()[i] != 1 ||
          ranges[i] == 1) {
        op.emitError("Unable to reduce dims to 3 for non-normalized loop");
        return failure();
      }
    }
    IVLOG(1, "Here we go!");
    // Initialize packing state
    SmallVector<int64_t, 3> curPack = {1, 1, 1};
    SmallVector<IndexPacking, 6> packInfo(ranges.size());
    // Pack all po2 indexes into index #0
    for (unsigned i = 0; i < ranges.size(); i++) {
      if (IsPo2(ranges[i])) {
        packInfo[i].sourceIdx = 0;
        packInfo[i].floorDiv = curPack[0];
        packInfo[i].mod = ranges[i];
        curPack[0] *= ranges[i];
      }
    }
    // Now pack all non-po2 into seperate indexes if possible (and spill the
    // rest to #2)
    unsigned curIdx = 0;
    for (unsigned i = 0; i < ranges.size(); i++) {
      if (!IsPo2(ranges[i])) {
        packInfo[i].sourceIdx = curIdx;
        packInfo[i].floorDiv = curPack[curIdx];
        packInfo[i].mod = ranges[i];
        curPack[curIdx] *= ranges[i];
        curIdx = std::min(curIdx + 1, 2u);
      }
    }
    IVLOG(1, "Past packing");
    // Remove extra indexes
    while (curPack.size() > 0 && curPack.back() == 1) {
      curPack.pop_back();
    }
    // Make a noralized affineParallel
    auto builder = OpBuilder(op);
    auto newLoop = builder.create<AffineParallelOp>(
        op.getLoc(),
        /*resultsTypes=*/ArrayRef<Type>{},
        /*reductions=*/ArrayRef<AtomicRMWKind>{},
        /*ranges=*/curPack);
    // Make an affine apply for each original index
    auto ibuild = newLoop.getBodyBuilder();
    for (unsigned i = 0; i < ranges.size(); i++) {
      const auto &info = packInfo[i];
      // Begin with the original value
      AffineExpr cur = getAffineDimExpr(0, op.getContext());
      // Divide by floorDiv (== 1 case should be canonicalized away)
      cur = cur.floorDiv(info.floorDiv);
      // If needed, do a modulus
      if (info.floorDiv * info.mod !=
          static_cast<uint64_t>(curPack[info.sourceIdx])) {
        cur = cur % info.mod;
      }
      // Make an affine map
      auto map = AffineMap::get(1, 0, cur);
      // Make the affine apply
      Value newIV = newLoop.getIVs()[info.sourceIdx];
      Value mapped =
          ibuild.create<AffineApplyOp>(op.getLoc(), map, ValueRange{newIV});
      op.getIVs()[i].replaceAllUsesWith(mapped);
    }
    IVLOG(1, "Past mapping: " << debugString(*newLoop.getOperation()));
    // Slice in the interior
    newLoop.getBody()->getOperations().splice( //
        std::prev(newLoop.getBody()->end()),   //
        op.getBody()->getOperations(),         //
        op.getBody()->begin(),                 //
        std::prev(op.getBody()->end()));
    newLoop.setAttr("hardware", hardware);
    IVLOG(1, "Past merging: " << debugString(*newLoop.getOperation()));
    // Erase old op
    op.erase();
    IVLOG(1, "Should be good");
    return success();
  }
  void runOnFunction() override {
    getFunction().walk([&](AffineParallelOp op) {
      if (failed(maybePack(op))) {
        signalPassFailure();
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createAffinePackPass() {
  return std::make_unique<AffinePackPass>();
}
} // namespace pmlc::target::intel_gen
