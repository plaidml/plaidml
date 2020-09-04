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
#include "pmlc/util/tags.h"

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
struct AffineIndexPackPass : public AffineIndexPackBase<AffineIndexPackPass> {
  LogicalResult maybePack(AffineParallelOp op) {
    bool isThread = hasUnitTag(op, gpuThreadTag());
    bool isBlock = hasUnitTag(op, gpuBlockTag());
    bool isHardware = (isBlock || isThread);
    auto subgroupSize = getIntegerTag(op, subgroupSizeTag(), 1);
    unsigned maxDims = (isThread && subgroupSize > 1) ? 2 : 3;
    if (!isHardware ||
        (op.getIVs().size() <= maxDims && op.getIVs().size() != 0)) {
      // Doesn't need special handling
      return success();
    }
    // Check for unsupported cases
    if (op.getNumResults() != 0) {
      op.emitError(
          "Unable to reduce dims for affine.parallel that produces results");
      return failure();
    }
    auto maybeRanges = op.getConstantRanges();
    if (!maybeRanges) {
      op.emitError("Unable to reduce dims for non-constant ranges");
      return failure();
    }
    auto ranges = *maybeRanges;
    // Verify outer loop is normalized and no trip-count 1 ranges
    for (unsigned i = 0; i < ranges.size(); i++) {
      auto lbExpr =
          op.lowerBoundsMap().getResult(i).dyn_cast<AffineConstantExpr>();
      if (!lbExpr || lbExpr.getValue() != 0 || op.getSteps()[i] != 1 ||
          ranges[i] == 1) {
        op.emitError("Unable to reduce dims for non-normalized loop");
        return failure();
      }
    }
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
        curIdx = std::min(curIdx + 1, maxDims - 1);
      }
    }
    // Remove extra indexes with range 1
    while (curPack.size() > 1 && curPack.back() == 1) {
      curPack.pop_back();
    }
    // Make a normalized affineParallel
    auto builder = OpBuilder(op);
    auto newLoop = builder.create<AffineParallelOp>(
        op.getLoc(),
        /*resultsTypes=*/ArrayRef<Type>{},
        /*reductions=*/ArrayRef<AtomicRMWKind>{},
        /*ranges=*/curPack);
    // Make an affine apply for each original index
    auto innerBuilder = newLoop.getBodyBuilder();
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
      // Convert the affine expression into a single result map
      auto map = AffineMap::get(1, 0, cur);
      // Apply the map to the new IV to generate the old IV
      Value newIV = newLoop.getIVs()[info.sourceIdx];
      Value mapped = innerBuilder.create<AffineApplyOp>(op.getLoc(), map,
                                                        ValueRange{newIV});
      op.getIVs()[i].replaceAllUsesWith(mapped);
    }
    // Splice in the interior
    newLoop.getBody()->getOperations().splice( //
        std::prev(newLoop.getBody()->end()),   //
        op.getBody()->getOperations(),         //
        op.getBody()->begin(),                 //
        std::prev(op.getBody()->end()));

    // Copy across any tags
    copyTags(newLoop, op);

    // Erase old op
    op.erase();
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

std::unique_ptr<mlir::Pass> createAffineIndexPackPass() {
  return std::make_unique<AffineIndexPackPass>();
}
} // namespace pmlc::target::intel_gen
