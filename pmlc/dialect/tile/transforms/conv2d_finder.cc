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
  explicit Conv2dFinder(ContractionOp op) : contract(op) {}

  bool isaConv2d() {
    if (contract.combo() != CombinationKind::mul) {
      reason = "Invalid CombinationKind";
      return false;
    }

    if (contract.agg() != AggregationKind::add) {
      reason = "Invalid AggregationKind";
      return false;
    }

    // assume activation is 0, kernel is 1
    assert(contract.getNumTensors() == 2);
    AffineMap act = contract.getSourceMap(0);
    AffineMap ker = contract.getSourceMap(1);
    assert(contract.getNumResults() == 1);
    AffineMap out = contract.sink();

    // assume N is dimension 0 of activation, output
    unsigned nPos = 0;
    if (act.getResult(nPos) != out.getResult(nPos)) {
      reason = "Unable to find batch dimension";
      return false;
    }

    // look for Ci
    // skip dimension 0 (nPos) of the activation
    unsigned actCiPos = 1;
    unsigned kerCiPos = 0;
    if (!findCommonResult(act, ker, actCiPos, kerCiPos)) {
      reason = "Unable to find channel-in dimension";
      return false;
    }

    // look for Co
    // skip dimension 0 (nPos) of the output
    unsigned outCoPos = 1;
    unsigned kerCoPos = 0;
    if (!findCommonResult(out, ker, outCoPos, kerCoPos)) {
      reason = "Unable to find channel-out dimension";
      return false;
    }

    // activation, kernel, output must be 4D tensors
    if (act.getNumResults() != numTotalDims ||
        ker.getNumResults() != numTotalDims ||
        out.getNumResults() != numTotalDims) {
      reason = "Invalid tensor rank";
      return false;
    }

    // calculate spatial dimensions
    llvm::SmallVector<unsigned, numSpatialDims> actSpatial, kerSpatial,
        outSpatial;
    for (unsigned i = 0; i < numTotalDims; ++i) {
      // activation - exclude batch and channel in dimensions
      if (i != nPos && i != actCiPos) {
        actSpatial.push_back(i);
      }
      // kernel - exclude channel in and out dimensions
      if (i != kerCiPos && i != kerCoPos) {
        kerSpatial.push_back(i);
      }
      // output - exclude batch and channel out dimensions
      if (i != nPos && i != outCoPos) {
        outSpatial.push_back(i);
      }
    }

    if (actSpatial.size() != numSpatialDims ||
        kerSpatial.size() != numSpatialDims ||
        outSpatial.size() != numSpatialDims) {
      reason = "Invalid spatial rank";
      return false;
    }

    // flatten the Affine expressions in each AffineMap
    std::vector<flatExpr> actFlatExprs, kerFlatExprs, outFlatExprs;
    if (failed(getFlattenedAffineExprs(act, &actFlatExprs, nullptr)) ||
        failed(getFlattenedAffineExprs(ker, &kerFlatExprs, nullptr)) ||
        failed(getFlattenedAffineExprs(out, &outFlatExprs, nullptr))) {
      reason = "Unable to get flattened Affine expressions";
      return false;
    }

    // Canonical NHWC example
    // N = batch
    // Y = output Y
    // X = output X
    // Co = Channel out
    // Ky = Kernel Y
    // Kx = Kernel X
    // Ci = Channel in
    // P = padding
    // S = strides
    // D = dilations
    //                 N Y X C K K C
    //                       o y x i
    // Activation:
    // (N)  flatExpr = 1 0 0 0 0 0 0 0
    // (H)  flatExpr = 0 S 0 0 D 0 0 P
    // (W)  flatExpr = 0 0 S 0 0 D 0 P
    // (Ci) flatExpr = 0 0 0 0 0 0 1 0
    // Kernel:
    // (H)  flatExpr = 0 0 0 0 1 0 0 0
    // (W)  flatExpr = 0 0 0 0 0 1 0 0
    // (Ci) flatExpr = 0 0 0 0 0 0 1 0
    // (Co) flatExpr = 0 0 0 1 0 0 0 0
    // Output:
    // (N)  flatExpr = 1 0 0 0 0 0 0 0
    // (H)  flatExpr = 0 1 0 0 0 0 0 0
    // (W)  flatExpr = 0 0 1 0 0 0 0 0
    // (Co) flatExpr = 0 0 0 1 0 0 0 0
    for (unsigned i = 0; i < numSpatialDims; ++i) {
      flatExpr actFlat = actFlatExprs[actSpatial[i]];
      flatExpr kerFlat = kerFlatExprs[kerSpatial[i]];
      flatExpr outFlat = outFlatExprs[outSpatial[i]];

      // Activation:
      // (H)  flatExpr = 0 S 0 0 D 0 0 P*
      // (W)  flatExpr = 0 0 S 0 0 D 0 P*
      int64_t padding = actFlat.back();

      // Activation:
      // (H)  flatExpr = 0 S* 0 0 D 0 0 P
      // (W)  flatExpr = 0 0 S* 0 0 D 0 P
      // Output:
      // (H)  flatExpr = 0 1* 0 0 0 0 0 0
      // (W)  flatExpr = 0 0 1* 0 0 0 0 0
      int64_t stride = multiply(actFlat, outFlat);

      // Activation:
      // (H)  flatExpr = 0 S 0 0 D* 0 0 P
      // (W)  flatExpr = 0 0 S 0 0 D* 0 P
      // Kernel:
      // (H)  flatExpr = 0 0 0 0 1* 0 0 0
      // (W)  flatExpr = 0 0 0 0 0 1* 0 0
      int64_t dilation = multiply(actFlat, kerFlat);

      // Expecting non-zero stride, dilation else something is wrong
      if (!stride || !dilation) {
        reason = "Invalid spatial dimensions";
        return false;
      }

      paddings.push_back(padding);
      strides.push_back(stride);
      dilations.push_back(dilation);
    }

    isConv2d = true;
    return true;
  }

  std::string getReason() {
    assert(!conv2d);
    return reason;
  }
  SmallVector<int64_t, 2> getPaddings() {
    assert(conv2d);
    return paddings;
  }
  SmallVector<int64_t, 2> getStrides() {
    assert(conv2d);
    return strides;
  }
  SmallVector<int64_t, 2> getDilations() {
    assert(conv2d);
    return dilations;
  }

private:
  ContractionOp contract;
  bool isConv2d{false};
  std::string reason;
  static const unsigned numTotalDims{4};
  static const unsigned numSpatialDims{2};
  SmallVector<int64_t, numSpatialDims> paddings;
  SmallVector<int64_t, numSpatialDims> strides;
  SmallVector<int64_t, numSpatialDims> dilations;
  typedef llvm::SmallVector<int64_t, 8> flatExpr;

  int64_t multiply(flatExpr flatExprA, flatExpr flatExprB) {
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
    unsigned init = indexB;
    for (; indexA <= mapA.getNumResults(); ++indexA) {
      for (indexB = init; indexB < mapB.getNumResults(); ++indexB) {
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
  llvm::errs() << "Testing : " << getFunction().getName() << "\n";
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
      llvm::errs() << conv2dFinder.getReason() << "\n";
    }
  });
}

} // namespace

std::unique_ptr<Pass> createConv2dFinderPass() {
  return std::make_unique<Conv2dFinderPass>();
}

} // namespace pmlc::dialect::tile
