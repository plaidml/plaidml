// Copyright 2019, Intel Corporation

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "llvm/Support/MathExtras.h"

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/padding.h"
#include "pmlc/dialect/tile/transforms/pass_detail.h"
#include "pmlc/util/ident.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/strides.h"

namespace pmlc::dialect::tile {

using namespace mlir; // NOLINT

namespace {

struct PadRangesPass : public PadRangesBase<PadRangesPass> {
  int64_t maxPowerOfTwo = 16;
  int64_t minPowerOfTwo = 8;
  int64_t maxIncreasePercent = 5;
  void runOnFunction() final;
};

void PadRangesPass::runOnFunction() {
  auto func = getFunction();
  func.walk([&](ContractionOp op) {
    // Skip some cases where the padding pass can't operate.
    if (op.getNumSymbols()) {
      op.emitRemark("padding cannot be run on symbolic contractions");
      return;
    }
    if (!op.lowerBounds() || !op.upperBounds()) {
      op.emitRemark("contraction bounds must be computed");
      return;
    }
    // Extract which dimensions of the contraction are which output dimension
    SmallVector<unsigned, 4> outIdx;
    unsigned outRank = op.getResultType().getRank();
    for (unsigned i = 0; i < outRank; i++) {
      auto outDim = op.sink().getResult(i).dyn_cast<AffineDimExpr>();
      if (!outDim) {
        op.emitRemark("Invalid output access");
        return;
      }
      outIdx[i] = outDim.getPosition();
    }
    // Verify bounds are simple
    unsigned indexCount = op.lowerBounds()->getNumResults();
    SmallVector<int64_t, 6> ranges;
    for (unsigned i = 0; i < indexCount; i++) {
      auto lowerConst =
          op.lowerBounds()->getResult(i).dyn_cast<AffineConstantExpr>();
      auto upperConst =
          op.upperBounds()->getResult(i).dyn_cast<AffineConstantExpr>();
      if (!lowerConst || lowerConst.getValue() != 0) {
        op.emitRemark("Invalid lower bound");
        return;
      }
      if (!upperConst) {
        op.emitRemark("Invalid lower bound");
        return;
      }
      ranges.push_back(upperConst.getValue() + 1);
    }
    // Get strides for output
    auto outStrides = util::computeStrideArray(op.getResultType(), op.sink());
    IVLOG(1, "outStrides = " << *outStrides);
    // Decide wthat to round up
    bool didChange = false;
    SmallVector<int64_t, 6> newRanges = ranges;
    for (unsigned i = 0; i < indexCount; i++) {
      if (outStrides->strides[i] == 0) {
        continue;
      }
      for (int64_t po2 = maxPowerOfTwo; po2 >= minPowerOfTwo; po2 /= 2) {
        int64_t roundedRange = llvm::divideCeil(ranges[i], po2) * po2;
        if (roundedRange == ranges[i]) {
          break;
        }
        int64_t increase = roundedRange - ranges[i];
        int64_t precIncrease = increase * 100 / ranges[i];
        IVLOG(1, "Orig: " << ranges[i] << ", rounded: " << roundedRange
                          << ", precIncrease = " << precIncrease);
        if (precIncrease < maxIncreasePercent) {
          newRanges[i] = roundedRange;
          didChange = true;
          break;
        }
      }
    }
    // Exit early if nothing is changing
    if (!didChange) {
      return;
    }
    IVLOG(1, "Making a brave new op");
    // Start building a new contraction to 'shrink'
    OpBuilder builder(op.getContext());
    builder.setInsertionPointAfter(op.getOperation());
    // Make shinking contraction
    auto ident = createIdentity(builder, op.getLoc(), AtomicRMWKind::assign,
                                op.getResultType().getElementType());
    auto idMap = AffineMap::getMultiDimIdentityMap(outRank, op.getContext());
    auto newOp = builder.create<ContractionOp>(
        op.getLoc(), op.getResultType(), ident, ArrayRef<Value>{op},
        AggregationKind::assign, CombinationKind::none, idMap,
        ArrayRef<AffineMap>{idMap},
        IntegerSet::getEmptySet(outRank, 0, op.getContext()),
        /*noReduce=*/true, "shrink");
    newOp.setLowerBounds(SmallVector<int64_t, 4>(outRank, 0));
    SmallVector<int64_t, 4> newOutSize;
    SmallVector<int64_t, 4> shinkUpperBounds;
    for (unsigned i = 0; i < outRank; i++) {
      newOutSize.push_back(newRanges[outIdx[i]]);
      shinkUpperBounds.push_back(ranges[outIdx[i]] - 1);
    }
    newOp.setUpperBounds(shinkUpperBounds);
    // Switch all uses to the new contracton
    op.getResult().replaceAllUsesExcept(
        newOp.getResult(), SmallPtrSet<Operation *, 1>{newOp.getOperation()});
    // Resize + rerange the output of the original op
    op.getResult().setType(
        RankedTensorType::get(newOutSize, op.getResultType().getElementType()));
    for (unsigned i = 0; i < indexCount; i++) {
      newRanges[i] -= 1;
    }
    op.setUpperBounds(newRanges);
  });
}

} // namespace

std::unique_ptr<Pass> createPadRangesPass() {
  return std::make_unique<PadRangesPass>();
}

} // namespace pmlc::dialect::tile
