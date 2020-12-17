// Copyright 2020, Intel Corporation

#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/pass_detail.h"

#include <numeric>

using namespace mlir;                // NOLINT
using namespace pmlc::dialect::stdx; // NOLINT

namespace pmlc::dialect::tile {

namespace {

struct ExpandReshapePass : public ExpandReshapeBase<ExpandReshapePass> {
  void runOnFunction() final;

  // Compute the number of indices needed
  int computeNumIdxs(ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape);
  // Expand in-order reshape operation
  void expandReshape(ReshapeOp reshapeOp);
  // Expand out-of-order reshape operation
  void expandOutOfOrderReshape(ReshapeOp reshapeOp);
};

// Compute the number of indices needed
int ExpandReshapePass::computeNumIdxs(ArrayRef<int64_t> srcShape,
                                      ArrayRef<int64_t> dstShape) {
  SmallVector<int64_t, 4> srcDims(srcShape.begin(), srcShape.end());
  SmallVector<int64_t, 4> dstDims(dstShape.begin(), dstShape.end());
  int pSrc = srcDims.size() - 1;
  int pDst = dstDims.size() - 1;
  int numIdxs = 0;
  while (pSrc >= 0 && pDst >= 0) {
    while (pSrc >= 0 && srcDims[pSrc] == 1) {
      --pSrc;
    }
    if (pSrc < 0) {
      break;
    }
    while (pDst >= 0 && dstDims[pDst] == 1) {
      --pDst;
    }
    if (pDst < 0) {
      break;
    }
    int64_t gcd = std::gcd(srcDims[pSrc], dstDims[pDst]);
    if (gcd == 1) {
      return -1;
    }
    ++numIdxs;
    srcDims[pSrc] /= gcd;
    dstDims[pDst] /= gcd;
  }
  while (pSrc >= 0) {
    if (srcDims[pSrc] > 1) {
      ++numIdxs;
    }
    --pSrc;
  }
  while (pDst >= 0) {
    if (dstDims[pDst] > 1) {
      ++numIdxs;
    }
    --pDst;
  }
  return numIdxs;
}

// Expand in-order reshape operation
void ExpandReshapePass::expandReshape(ReshapeOp reshapeOp) {
  auto src = reshapeOp.tensor();
  auto srcShape = src.getType().cast<RankedTensorType>().getShape();
  auto dst = reshapeOp.getResult();
  auto dstType = dst.getType().cast<RankedTensorType>();
  auto dstShape = dstType.getShape();

  int numIdxs = computeNumIdxs(srcShape, dstShape);
  if (numIdxs < 0) {
    // The indices of src and dst are out-of-order
    expandOutOfOrderReshape(reshapeOp);
  }

  auto context = reshapeOp.getContext();
  SmallVector<int64_t, 4> srcDims(srcShape.begin(), srcShape.end());
  SmallVector<int64_t, 4> dstDims(dstShape.begin(), dstShape.end());
  SmallVector<int64_t, 4> ranges;
  auto zeroExpr = getAffineConstantExpr(0, context);
  SmallVector<AffineExpr, 4> srcExprs(srcDims.size(), zeroExpr);
  SmallVector<AffineExpr, 4> dstExprs(dstDims.size(), zeroExpr);

  int pSrc = srcDims.size() - 1;
  int pDst = dstDims.size() - 1;
  int64_t mulSrc = 1;
  int64_t mulDst = 1;
  while (pSrc >= 0 && pDst >= 0) {
    while (pSrc >= 0 && srcDims[pSrc] == 1) {
      --pSrc;
      mulSrc = 1;
    }
    if (pSrc < 0) {
      break;
    }
    while (pDst >= 0 && dstDims[pDst] == 1) {
      --pDst;
      mulDst = 1;
    }
    if (pDst < 0) {
      break;
    }
    int64_t gcd = std::gcd(srcDims[pSrc], dstDims[pDst]);
    assert(gcd > 1);
    int idxId = numIdxs - ranges.size() - 1;
    assert(idxId >= 0);
    ranges.emplace_back(gcd);
    srcExprs[pSrc] = srcExprs[pSrc] + getAffineConstantExpr(mulSrc, context) *
                                          getAffineDimExpr(idxId, context);
    dstExprs[pDst] = dstExprs[pDst] + getAffineConstantExpr(mulDst, context) *
                                          getAffineDimExpr(idxId, context);
    srcDims[pSrc] /= gcd;
    dstDims[pDst] /= gcd;
    mulSrc *= gcd;
    mulDst *= gcd;
  }

  while (pSrc >= 0) {
    if (srcDims[pSrc] > 1) {
      unsigned idxId = numIdxs - ranges.size() - 1;
      assert(idxId >= 0);
      ranges.emplace_back(srcDims[pSrc]);
      srcExprs[pSrc] = srcExprs[pSrc] + getAffineConstantExpr(mulSrc, context) *
                                            getAffineDimExpr(idxId, context);
    }
    --pSrc;
    mulSrc = 1;
  }
  while (pDst >= 0) {
    if (dstDims[pDst] > 1) {
      unsigned idxId = numIdxs - ranges.size() - 1;
      assert(idxId >= 0);
      ranges.emplace_back(dstDims[pDst]);
      dstExprs[pDst] = dstExprs[pDst] + getAffineConstantExpr(mulDst, context) *
                                            getAffineDimExpr(idxId, context);
    }
    --pDst;
    mulDst = 1;
  }

  AffineMap srcMap = AffineMap::get(numIdxs, 0, srcExprs, context);
  AffineMap sinkMap = AffineMap::get(numIdxs, 0, dstExprs, context);

  // Replace the original reshape with constraction
  OpBuilder builder(reshapeOp);
  auto elementType = dst.getType().cast<RankedTensorType>().getElementType();
  auto ident = tile::createIdentity(builder, reshapeOp.getLoc(), elementType,
                                    AggregationKind::assign);
  auto res = builder.create<ContractionOp>(
      reshapeOp.getLoc(),
      /* resultType = */ dst.getType(),
      /* init = */ ident,
      /* tensors = */ ArrayRef{src},
      /* agg = */ util::AggregationKind::assign,
      /* combo = */ util::CombinationKind::none,
      /* sink = */ sinkMap,
      /* srcs = */ ArrayRef{srcMap},
      /* cons = */ IntegerSet::getEmptySet(dstType.getRank(), 0, context),
      /* name = */ "reshape");
  res.setUpperBounds(ranges);
  dst.replaceAllUsesWith(res);
  reshapeOp.erase();
}

// Expand out-of-order reshape operation
void ExpandReshapePass::expandOutOfOrderReshape(ReshapeOp reshapeOp) { return; }

void ExpandReshapePass::runOnFunction() {
  auto func = getFunction();
  for (auto op : func.getOps<ReshapeOp>()) {
    expandReshape(op);
  }
  return;
}

} // namespace

std::unique_ptr<Pass> createExpandReshapePass() {
  return std::make_unique<ExpandReshapePass>();
}

} // namespace pmlc::dialect::tile
