// Copyright 2022 Intel Corporation

#include <memory>

#include "llvm/ADT/SmallBitVector.h"

#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "pmlc/dialect/linalgx/analysis/convolution.h"
#include "pmlc/dialect/linalgx/transforms/pass_detail.h"
#include "pmlc/dialect/linalgx/transforms/regulate_depthwise.h"

#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::linalgx {

struct RegulateDepthwisePass
    : public RegulateDepthwiseBase<RegulateDepthwisePass> {
  RegulateDepthwisePass() = default;
  void runOnOperation() final {
    auto func = getOperation();
    func.walk([&](linalg::GenericOp op) { regulateDepthwise(op); });
  }

  void joinAffineMapResult(SmallVector<AffineExpr, 12> &results,
                           AffineMap origMap, AffineMap newMap) {
    auto zero = getAffineConstantExpr(0, origMap.getContext());
    for (unsigned i = 0; i < newMap.getNumResults(); ++i) {
      results.emplace_back(newMap.getResult(i) == zero &&
                                   origMap.getResult(i) != zero
                               ? origMap.getResult(i)
                               : newMap.getResult(i));
    }
  }

  SmallVector<AffineMap, 3> simplifyMaps(ArrayRef<int64_t> ranges,
                                         AffineMap inputMap,
                                         AffineMap filterMap,
                                         AffineMap outputMap,
                                         llvm::SmallBitVector &usedDims) {
    size_t origNumDims = ranges.size();
    auto context = inputMap.getContext();
    SmallVector<AffineExpr, 8> rangeValues;
    // Find out the zero dimensions
    for (unsigned i = 0; i < origNumDims; ++i) {
      rangeValues.emplace_back(ranges[i] == 1
                                   ? getAffineConstantExpr(0, context)
                                   : getAffineDimExpr(i, context));
    }

    AffineMap valueMap = AffineMap::get(origNumDims, 0, rangeValues, context);
    AffineMap tmpInputMap = inputMap.compose(valueMap);
    AffineMap tmpFilterMap = filterMap.compose(valueMap);
    AffineMap tmpOutputMap = outputMap.compose(valueMap);
    SmallVector<AffineExpr, 12> results;

    // Join input/filter/output maps together.
    // Meanwhile, do not replace with single-dim expression with zero.
    joinAffineMapResult(results, inputMap, tmpInputMap);
    joinAffineMapResult(results, filterMap, tmpFilterMap);
    joinAffineMapResult(results, outputMap, tmpOutputMap);
    AffineMap tmpJoinedMap = AffineMap::get(origNumDims, 0, results, context);

    // Find out all used dims
    usedDims.resize(tmpJoinedMap.getNumDims());
    tmpJoinedMap.walkExprs([&](AffineExpr expr) {
      if (auto dim = expr.dyn_cast<AffineDimExpr>()) {
        usedDims.set(dim.getPosition());
      }
    });
    AffineMap joinedMap = compressUnusedDims(tmpJoinedMap);

    auto newNumDims = joinedMap.getNumDims();
    auto newResults = joinedMap.getResults();
    SmallVector<AffineExpr, 4> inputResults = {newResults[0], newResults[1],
                                               newResults[2], newResults[3]};
    AffineMap newInputMap =
        AffineMap::get(newNumDims, 0, inputResults, context);
    SmallVector<AffineExpr, 4> filterResults = {newResults[4], newResults[5],
                                                newResults[6], newResults[7]};
    AffineMap newFilterMap =
        AffineMap::get(newNumDims, 0, filterResults, context);
    SmallVector<AffineExpr, 4> outputResults = {newResults[8], newResults[9],
                                                newResults[10], newResults[11]};
    AffineMap newOutputMap =
        AffineMap::get(newNumDims, 0, outputResults, context);
    return {newInputMap, newFilterMap, newOutputMap};
  }

  SmallVector<AffineMap, 3> reorderMaps(SmallVector<AffineMap, 3> &origMaps,
                                        SmallVector<unsigned> &order) {
    if (util::dimPosition(origMaps[0].getResult(3)) == 3) {
      order = {0, 1, 2, 3, 6, 4, 5};
      auto context = origMaps[0].getContext();
      SmallVector<AffineExpr> results;
      for (unsigned d : order) {
        results.emplace_back(getAffineDimExpr(d, context));
      }
      AffineMap orderMap =
          AffineMap::get(origMaps[0].getNumDims(), origMaps[0].getNumSymbols(),
                         results, context);
      return {origMaps[0].compose(orderMap), origMaps[1].compose(orderMap),
              origMaps[2].compose(orderMap)};
    }
    order = {0, 1, 2, 3, 4, 5, 6};
    return origMaps;
  }

  void regulateDepthwise(linalg::GenericOp op) {
    Optional<ConvCapture> conv = detectConv(op);
    if (!conv) {
      IVLOG(3, "Cannot reorder: not a convolution. " << debugString(op));
      return;
    }

    // check that this is a 2D convolution
    if (conv->input.idxMap.getNumResults() != 4 ||
        conv->filter.idxMap.getNumResults() != 4 ||
        conv->output.idxMap.getNumResults() != 4) {
      IVLOG(1,
            "Cannot reorder: expected 2D convolution. op: " << debugString(op));
      return;
    }

    if (conv->input.idxMap.getNumDims() <= 7 ||
        conv->filter.idxMap.getNumDims() <= 7 ||
        conv->output.idxMap.getNumDims() <= 7) {
      IVLOG(1, "Nothing to be simplified.");
      return;
    }

    Optional<SmallVector<int64_t, 4>> ranges = op.getStaticLoopRanges();
    if (!ranges) {
      IVLOG(1, "Cannot reorder: expected static ranges.");
      return;
    }

    llvm::SmallBitVector usedDims;
    auto tmpMaps =
        simplifyMaps(*ranges, conv->input.idxMap, conv->filter.idxMap,
                     conv->output.idxMap, usedDims);
    if (usedDims.all()) {
      IVLOG(1, "Nothing to be simplified.");
      return;
    }
    unsigned numDims = 7;
    if (tmpMaps[0].getNumDims() != numDims ||
        tmpMaps[1].getNumDims() != numDims ||
        tmpMaps[2].getNumDims() != numDims) {
      IVLOG(1, "Cannot regulate: number of indexes is not 7.");
      return;
    }

    // check that we have (channels-last logical ordering):
    // (n, h, w, c), (r, s, c, c') -> (n, h, w, c)
    if (tmpMaps[0].getResult(3) != tmpMaps[1].getResult(2) ||
        tmpMaps[0].getResult(3) != tmpMaps[2].getResult(3)) {
      IVLOG(1, "Cannot regulate: expected channels-last logical ordering.");
      return;
    }

    SmallVector<unsigned> order;
    auto newMaps = reorderMaps(tmpMaps, order);
    AffineMap inputMap = newMaps[0];
    AffineMap filterMap = newMaps[1];
    AffineMap outputMap = newMaps[2];

    auto iterTypes = op.iterator_types().getValue();
    SmallVector<StringRef, 8> tmpIterTypes;
    SmallVector<int64_t, 8> tmpRanges;
    for (unsigned i = 0; i < iterTypes.size(); ++i) {
      if (usedDims[i]) {
        tmpIterTypes.emplace_back(iterTypes[i].cast<StringAttr>().getValue());
        tmpRanges.emplace_back((*ranges)[i]);
      }
    }
    SmallVector<StringRef, 8> newIterTypes(numDims);
    SmallVector<int64_t, 8> newRanges(numDims);
    for (unsigned i = 0; i < numDims; ++i) {
      newIterTypes[order[i]] = tmpIterTypes[i];
      newRanges[order[i]] = tmpRanges[i];
    }

    ImplicitLocOpBuilder builder(op->getLoc(), op);
    auto newOp = builder.create<linalg::GenericOp>(
        /*resultTensorTypes=*/TypeRange{conv->output.type},
        /*inputs=*/ValueRange{conv->input.value, conv->filter.value},
        /*outputs=*/ValueRange{conv->output.value},
        /*indexingMaps=*/ArrayRef<AffineMap>{inputMap, filterMap, outputMap},
        /*iteratorTypes=*/newIterTypes,
        /*doc=*/"",
        /*libraryCall=*/"",
        [](OpBuilder &builder, Location loc, ValueRange args) {
          auto mul = builder.create<arith::MulFOp>(loc, args[0], args[1]);
          auto add = builder.create<arith::AddFOp>(loc, args[2], mul);
          builder.create<linalg::YieldOp>(loc, ValueRange{add});
        });
    newOp->setAttr("iterator_ranges", builder.getI64ArrayAttr(newRanges));
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      op.getResult(i).replaceAllUsesWith(newOp.getResult(i));
    }
    op.erase();
  }
};

std::unique_ptr<mlir::Pass> createRegulateDepthwisePass() {
  return std::make_unique<RegulateDepthwisePass>();
}

} // namespace pmlc::dialect::linalgx
