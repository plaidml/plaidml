//
// Created by cxh on 8/6/20.
//
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Pass/Pass.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "pmlc/dialect/pxa/transforms/tile.h"

#include "mlir/Support/DebugStringHelper.h"
#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT
using mlir::AffineParallelOp;

namespace pmlc::dialect::pxa {

namespace {
using llvm::DenseMap;
using llvm::DenseSet;
using llvm::SmallVector;
using mlir::BlockArgument;

Operation *GetOriginalDef(Value val) {
  auto opRes = val.cast<mlir::OpResult>();
  while (true) {
    auto ap = mlir::dyn_cast<AffineParallelOp>(opRes.getOwner());
    if (!ap)
      break;
    auto ret = mlir::cast<AffineYieldOp>(ap.getBody()->getTerminator());
    auto src = ret.getOperand(opRes.getResultNumber());
    auto defop = src.getDefiningOp();
    if (dyn_cast<AffineReduceOp>(defop)) {
      opRes = src.cast<mlir::OpResult>();
    } else if (dyn_cast<AffineIfOp>(defop)) {
      defop->walk([&](AffineReduceOp op) {
        opRes = op.getResult().cast<mlir::OpResult>();
      });
    }
  }
  return opRes.getOwner();
}

void accelerate() {}

bool isAggregation(AffineParallelOp op) {
  bool AggTag = false;
  auto argRange = op.getIVs();
  auto parallelLoopNum = argRange.size() < 2 ? argRange.size() : 2;
  op.walk([&](AffineReduceOp reduce) {
    auto range = reduce.idxs();
    for (size_t i = 0; i < parallelLoopNum; i++) {
      auto firstArg = std::find(range.begin(), range.end(), argRange[i]);
      AggTag = firstArg == range.end() || AggTag;
    }
  });
  IVLOG(1, "the aggtag is " << AggTag);
  return AggTag;
}

void TileAccumulations(AffineParallelOp op) {
  // Find the originating reduce
  assert(op.getNumResults() == 1);
  if (isAggregation(op)) {
    auto srcDef = GetOriginalDef(op.getResult(0));
    auto red = mlir::dyn_cast<AffineReduceOp>(srcDef);
    // Get strides for output
    auto si = *computeStrideInfo(red);
    // Find all the accumulation indexes (stride 0 with respect to output) and
    // tile them into an inner block
    auto ranges = *op.getConstantRanges();
    SmallVector<int64_t, 6> accumTile;
    auto steps = op.steps().cast<ArrayAttr>().getValue();
    for (size_t i = 0; i < ranges.size(); i++) {
      auto arg = op.getIVs()[i];
      if (si.strides.count(arg)) {
        accumTile.push_back(steps[i].cast<IntegerAttr>().getInt());
      } else {
        accumTile.push_back(ranges[i]);
      }
      IVLOG(1, "accumTile[" << i << "] = " << accumTile[i]);
    }
    performTiling(op, accumTile);
  } else {
    // TODO maybe accelerate the loop in one kernel.
    accelerate();
  }
}

struct TileAccumulatePass : public TileAccumulateBase<TileAccumulatePass> {
  void runOnFunction() final {
    auto func = getFunction();
    // Autotile only the outermost loops
    for (auto &op : func.getBody().front()) {
      auto loop = mlir::dyn_cast<mlir::AffineParallelOp>(op);
      if (!loop) {
        continue;
      }
      auto ranges = loop.getConstantRanges();
      if (!ranges) {
        return;
      }
      TileAccumulations(loop);
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createTileAccumulatePass() {
  return std::make_unique<TileAccumulatePass>();
}

} // namespace pmlc::dialect::pxa.
