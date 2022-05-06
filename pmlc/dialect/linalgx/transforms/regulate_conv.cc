// Copyright 2022 Intel Corporation

#include <memory>

#include "llvm/ADT/SmallBitVector.h"

#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/linalgx/analysis/convolution.h"
#include "pmlc/dialect/linalgx/transforms/pass_detail.h"
#include "pmlc/dialect/linalgx/transforms/regulate_conv.h"

#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::linalgx {

struct RegulateConvolutionPass
    : public RegulateConvolutionBase<RegulateConvolutionPass> {
  RegulateConvolutionPass() = default;
  void runOnFunction() final {
    auto func = getFunction();
    func.walk([&](linalg::GenericOp op) { regulateConvolution(op); });
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

  void regulateConvolution(linalg::GenericOp op) {
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
    auto newMaps =
        simplifyMaps(*ranges, conv->input.idxMap, conv->filter.idxMap,
                     conv->output.idxMap, usedDims);
    if (usedDims.all()) {
      IVLOG(1, "Nothing to be simplified.");
      return;
    }
    AffineMap inputMap = newMaps[0];
    AffineMap filterMap = newMaps[1];
    AffineMap outputMap = newMaps[2];
    if (inputMap.getNumDims() != 7 || filterMap.getNumDims() != 7 ||
        outputMap.getNumDims() != 7) {
      IVLOG(1, "Cannot reorder: number of indexes is not 7.");
      return;
    }

    // check that we have (channels-last logical ordering):
    // (n, h, w, c), (r, s, c, k) -> (n, h, w, k)
    if (inputMap.getResult(3) != filterMap.getResult(2)) {
      IVLOG(1, "Cannot reorder: expected channels-last logical ordering.");
      return;
    }
    if (filterMap.getResult(3) != outputMap.getResult(3) &&
        filterMap.getResult(3) ==
            getAffineConstantExpr(0, filterMap.getContext())) {
      IVLOG(1, "Cannot reorder: expected channels-last logical ordering.");
      return;
    }

    auto iterTypes = op.iterator_types().getValue();
    SmallVector<StringRef, 8> newIterTypes;
    SmallVector<int64_t, 8> newRanges;
    for (unsigned i = 0; i < iterTypes.size(); ++i) {
      if (usedDims[i]) {
        newIterTypes.emplace_back(iterTypes[i].cast<StringAttr>().getValue());
        newRanges.emplace_back((*ranges)[i]);
      }
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
          auto mul = builder.create<MulFOp>(loc, args[0], args[1]);
          auto add = builder.create<AddFOp>(loc, args[2], mul);
          builder.create<linalg::YieldOp>(loc, ValueRange{add});
        });
    newOp->setAttr("iterator_ranges", builder.getI64ArrayAttr(newRanges));
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      op.getResult(i).replaceAllUsesWith(newOp.getResult(i));
    }
    op.erase();
  }
};

std::unique_ptr<mlir::Pass> createRegulateConvolutionPass() {
  return std::make_unique<RegulateConvolutionPass>();
}

} // namespace pmlc::dialect::linalgx
