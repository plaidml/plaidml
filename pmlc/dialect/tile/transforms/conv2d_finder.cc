// Copyright 2021, Intel Corporation

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "llvm/Support/MathExtras.h"

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/padding.h"
#include "pmlc/dialect/tile/transforms/pass_detail.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/math/util.h"
#include "pmlc/util/strides.h"

namespace pmlc::dialect::tile {

using namespace mlir; // NOLINT

namespace {

struct Conv2dFinderPass : public Conv2dFinderBase<Conv2dFinderPass> {
  void runOnFunction() final;
};

class Conv2dFinder {
public:
  explicit Conv2dFinder(ContractionOp op) {
    // must be 2 inputs with comb=mul and agg=add
    if (op.combo() != CombinationKind::mul ||
        op.agg() != AggregationKind::add || op.getNumTensors() != 2) {
      isConv2d = false;
      return;
    }

    // TODO
    // must not have constraints
    if (auto constraints = *op.cons()) {
      if (constraints.getNumConstraints() != 0) {
        isConv2d = false;
        return;
      }
    }

    // assume activation is 0, kernel is 1
    AffineMap act = op.getSourceMap(0);
    AffineMap ker = op.getSourceMap(1);
    AffineMap out = op.sink();

    // activation, kernel, output must be 4D tensors
    if (act.getNumResults() != numDims || ker.getNumResults() != numDims ||
        out.getNumResults() != numDims) {
      isConv2d = false;
      return;
    }

    // assume N is dimension 0 of activation, output
    unsigned nPos = 0;
    if (act.getResult(nPos) != out.getResult(nPos)) {
      isConv2d = false;
      return;
    }

    // look for Co
    unsigned outCoPos = 0;
    unsigned kerCoPos = 0;
    if (!findCommonResult(out, ker, outCoPos, kerCoPos)) {
      isConv2d = false;
      return;
    }

    // look for Ci
    unsigned actCiPos = 0;
    unsigned kerCiPos = 0;
    if (!findCommonResult(act, ker, actCiPos, kerCiPos)) {
      isConv2d = false;
      return;
    }

    // calculate spatial dimensions
    llvm::SmallVector<unsigned, numSpatialDims> actSpatial;
    for (unsigned i = 0; i < numDims; ++i) {
      if (i != nPos && i != actCiPos) {
        actSpatial.push_back(i);
      }
    }

    llvm::SmallVector<unsigned, numSpatialDims> kerSpatial;
    for (unsigned i = 0; i < numDims; ++i) {
      if (i != kerCiPos && i != kerCoPos) {
        kerSpatial.push_back(i);
      }
    }

    llvm::SmallVector<unsigned, numSpatialDims> outSpatial;
    for (unsigned i = 0; i < numDims; ++i) {
      if (i != nPos && i != outCoPos) {
        outSpatial.push_back(i);
      }
    }

    if (actSpatial.size() != numSpatialDims ||
        kerSpatial.size() != numSpatialDims ||
        outSpatial.size() != numSpatialDims) {
      isConv2d = false;
      return;
    }

    // flatten the Affine expressions
    std::vector<llvm::SmallVector<int64_t, 8>> actFlatExprs;
    if (failed(getFlattenedAffineExprs(act, &actFlatExprs, nullptr))) {
      isConv2d = false;
      return;
    }

    std::vector<llvm::SmallVector<int64_t, 8>> kerFlatExprs;
    if (failed(getFlattenedAffineExprs(ker, &kerFlatExprs, nullptr))) {
      isConv2d = false;
      return;
    }

    std::vector<llvm::SmallVector<int64_t, 8>> outFlatExprs;
    if (failed(getFlattenedAffineExprs(out, &outFlatExprs, nullptr))) {
      isConv2d = false;
      return;
    }

    // assume spatial dimensions are ordered between tensors
    // i.e. can't have NHWC output and NWHC input
    //      H and W will appear in the same order
    for (auto dim : actSpatial) {
      int64_t padding = actFlatExprs[dim].back();
      paddings.push_back(padding);
    }

    for (unsigned i = 0; i < numSpatialDims; ++i) {
      int64_t dilation =
          multiply(actFlatExprs[actSpatial[i]], kerFlatExprs[kerSpatial[i]]);
      if (!dilation) {
        isConv2d = false;
        return;
      }
      dilations.push_back(dilation);
    }

    for (unsigned i = 0; i < numSpatialDims; ++i) {
      int64_t stride =
          multiply(actFlatExprs[actSpatial[i]], outFlatExprs[outSpatial[i]]);
      if (!stride) {
        isConv2d = false;
        return;
      }
      strides.push_back(stride);
    }

    isConv2d = true;
    return;
  }

  bool isaConv2d() { return isConv2d; }
  SmallVector<int64_t, 2> getPaddings() {
    assert(isConv2d);
    return paddings;
  }
  SmallVector<int64_t, 2> getDilations() {
    assert(isConv2d);
    return dilations;
  }
  SmallVector<int64_t, 2> getStrides() {
    assert(isConv2d);
    return strides;
  }

private:
  bool isConv2d{false};
  static const unsigned numDims{4};
  static const unsigned numSpatialDims{2};
  SmallVector<int64_t, numSpatialDims> paddings;
  SmallVector<int64_t, numSpatialDims> dilations;
  SmallVector<int64_t, numSpatialDims> strides;

  int64_t multiply(llvm::SmallVector<int64_t, 8> flatExprA,
                   llvm::SmallVector<int64_t, 8> flatExprB) {
    size_t size = flatExprA.size();
    assert(flatExprB.size() == size);
    int64_t ret = 0;
    for (size_t i = 0; i < size; ++i) {
      ret += (flatExprA[i] * flatExprB[i]);
    }
    return ret;
  }

  bool findCommonResult(AffineMap mapA, AffineMap mapB, unsigned &indexA,
                        unsigned &indexB) {
    for (indexA = 0; indexA <= mapA.getNumResults(); ++indexA) {
      for (indexB = 0; indexB < mapB.getNumResults(); ++indexB) {
        if (mapA.getResult(indexA) == mapB.getResult(indexB)) {
          return true;
        }
      }
    }
    return false;
  }
};

void Conv2dFinderPass::runOnFunction() {
  auto func = getFunction();
  func.walk([&](ContractionOp op) {
    Conv2dFinder conv2dFinder(op);
    if (conv2dFinder.isaConv2d()) {
      llvm::errs() << "You say you want a convolution.\n";

      llvm::errs() << "paddings =";
      auto paddings = conv2dFinder.getPaddings();
      for (size_t i = 0; i < paddings.size(); ++i) {
        llvm::errs() << " " << paddings[i];
      }
      llvm::errs() << "\n";

      llvm::errs() << "strides =";
      auto strides = conv2dFinder.getStrides();
      for (size_t i = 0; i < strides.size(); ++i) {
        llvm::errs() << " " << strides[i];
      }
      llvm::errs() << "\n";

      llvm::errs() << "dilations =";
      auto dilations = conv2dFinder.getDilations();
      for (size_t i = 0; i < dilations.size(); ++i) {
        llvm::errs() << " " << dilations[i];
      }
      llvm::errs() << "\n";

    } else {
      llvm::errs() << "Well, you know, we all want to change the world.\n";
    }
  });
}

} // namespace

std::unique_ptr<Pass> createConv2dFinderPass() {
  return std::make_unique<Conv2dFinderPass>();
}

} // namespace pmlc::dialect::tile
