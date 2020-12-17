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

  void expandReshape(ReshapeOp reshapeOp);
  void expandOutOfOrderReshape(ReshapeOp reshapeOp);
};

void ExpandReshapePass::expandReshape(ReshapeOp reshapeOp) {
  auto src = reshapeOp.tensor();
  auto srcShape = src.getType().cast<RankedTensorType>().getShape();
  SmallVector<int64_t, 4> srcDims(srcShape.begin(), srcShape.end());
  auto dst = reshapeOp.getResult();
  auto dstType = dst.getType().cast<RankedTensorType>();
  auto dstShape = dstType.getShape();
  SmallVector<int64_t, 4> dstDims(dstShape.begin(), dstShape.end());

  auto context = reshapeOp.getContext();
  SmallVector<int64_t, 4> idxs;
  auto zeroExpr = getAffineConstantExpr(0, context);
  SmallVector<AffineExpr, 4> srcExprs(srcDims.size(), zeroExpr);
  SmallVector<AffineExpr, 4> dstExprs(dstDims.size(), zeroExpr);

  int pSrc = srcDims.size() - 1;
  int pDst = dstDims.size() - 1;
  int64_t mulSrc = 1;
  int64_t mulDst = 1;
  while (pSrc >= 0 && pDst >=0) {
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
    if (gcd == 1) {
      // The indices of src and dst are out-of-order
      expandOutOfOrderReshape(reshapeOp);
      return;
    }
    unsigned idxId = idxs.size();
    idxs.emplace_back(gcd);
    srcExprs[pSrc] = srcExprs[pSrc] + getAffineConstantExpr(mulSrc, context) * getAffineDimExpr(idxId, context);
    dstExprs[pDst] = dstExprs[pDst] + getAffineConstantExpr(mulDst, context) * getAffineDimExpr(idxId, context);
    srcDims[pSrc] /= gcd;
    dstDims[pDst] /= gcd;
    mulSrc *= gcd;
    mulDst *= gcd;
  } 

  while (pSrc >= 0) {
    if (srcDims[pSrc] > 1) {
      unsigned idxId = idxs.size();
      idxs.emplace_back(srcDims[pSrc]);
      srcExprs[pSrc] = srcExprs[pSrc] + getAffineConstantExpr(mulSrc, context) * getAffineDimExpr(idxId, context);
    }
    --pSrc;
    mulSrc = 1;
  }
  while (pDst >= 0) {
    if (dstDims[pDst] > 1) {
      unsigned idxId = idxs.size();
      idxs.emplace_back(dstDims[pDst]);
      dstExprs[pDst] = dstExprs[pDst] + getAffineConstantExpr(mulDst, context) * getAffineDimExpr(idxId, context);
    }
    --pDst;
    mulDst = 1;
  }

  // Reverse the dim indices
  DenseMap<AffineExpr, AffineExpr> map;
  unsigned numIdxs = idxs.size();
  for (unsigned i = 0; i < numIdxs / 2; ++i) {
    auto expr0 = getAffineDimExpr(i, context);
    auto expr1 = getAffineDimExpr(numIdxs - i - 1, context);
    map[expr0] = expr1;
    map[expr1] = expr0;
  }
  for (unsigned i = 0; i < srcExprs.size(); ++i) {
    srcExprs[i] = srcExprs[i].replace(map);
  }
  for (unsigned i = 0; i < dstExprs.size(); ++i) {
    dstExprs[i] = dstExprs[i].replace(map);
  }

  AffineMap srcMap = AffineMap::get(idxs.size(), 0, srcExprs, context);
  AffineMap sinkMap = AffineMap::get(idxs.size(), 0, dstExprs, context);

  OpBuilder builder(reshapeOp);
  auto elementType = dst.getType().cast<RankedTensorType>().getElementType();
  auto ident = tile::createIdentity(builder, reshapeOp.getLoc(), elementType, AggregationKind::assign);
  auto res = builder.create<ContractionOp>(reshapeOp.getLoc(),
    /* resultType = */ dst.getType(),
    /* init = */       ident,
    /* tensors = */    ArrayRef{src},
    /* agg = */        util::AggregationKind::assign,
    /* combo = */      util::CombinationKind::none,
    /* sink = */       sinkMap,
    /* srcs = */       ArrayRef{srcMap},
    /* cons = */       IntegerSet::getEmptySet(dstType.getRank(), 0, context),
    /* name = */       "reshape");
  dst.replaceAllUsesWith(res);
  reshapeOp.erase();
}

void ExpandReshapePass::expandOutOfOrderReshape(ReshapeOp reshapeOp) {
  // TODO: expand reshape with out-of-order indices
  return;
}

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
