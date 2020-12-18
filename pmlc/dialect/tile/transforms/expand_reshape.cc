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

// Compute the number of indices needed
int computeNumIdxs(ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape) {
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

Value flattenTensor(OpBuilder &builder, Value src) {
  auto context = builder.getContext();
  auto srcType = src.getType().cast<RankedTensorType>();
  auto shape = srcType.getShape();
  unsigned numIdxs = shape.size();
  int64_t size = 1;

  // Compute the Affine exprs and maps
  SmallVector<AffineExpr, 4> srcExprs(shape.size());
  AffineExpr dstExpr = getAffineConstantExpr(0, context);
  for (int dim = shape.size() - 1; dim >= 0; --dim) {
    srcExprs[dim] = getAffineDimExpr(dim, context);
    dstExpr = dstExpr + getAffineConstantExpr(size, context) *
                            getAffineDimExpr(dim, context);
    size *= shape[dim];
  }
  AffineMap srcMap = AffineMap::get(numIdxs, 0, srcExprs, context);
  AffineMap sinkMap = AffineMap::get(numIdxs, 0, dstExpr);

  // Allocate the linear tensor
  auto elementType = srcType.getElementType();
  auto dstType = RankedTensorType::get({size}, elementType);
  auto ident = tile::createIdentity(builder, builder.getUnknownLoc(),
                                    elementType, AggregationKind::assign);
  return builder.create<ContractionOp>(
      builder.getUnknownLoc(),
      /* resultType = */ dstType,
      /* init = */ ident,
      /* tensors = */ ArrayRef{src},
      /* agg = */ util::AggregationKind::assign,
      /* combo = */ util::CombinationKind::none,
      /* sink = */ sinkMap,
      /* srcs = */ ArrayRef{srcMap},
      /* cons = */ IntegerSet::getEmptySet(dstType.getRank(), 0, context),
      /* name = */ "flatten");
}

Value reshapeTensor(OpBuilder &builder, Value src, ArrayRef<int64_t> dstShape) {
  auto srcShape = src.getType().cast<RankedTensorType>().getShape();

  int numIdxs = computeNumIdxs(srcShape, dstShape);
  if (numIdxs < 0) {
    // The indices of src and dst are out-of-order
    return Value();
  }

  auto context = builder.getContext();
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
  auto elementType = src.getType().cast<RankedTensorType>().getElementType();
  auto dstType = RankedTensorType::get(dstShape, elementType);
  auto ident = tile::createIdentity(builder, builder.getUnknownLoc(),
                                    elementType, AggregationKind::assign);
  return builder.create<ContractionOp>(
      builder.getUnknownLoc(),
      /* resultType = */ dstType,
      /* init = */ ident,
      /* tensors = */ ArrayRef{src},
      /* agg = */ util::AggregationKind::assign,
      /* combo = */ util::CombinationKind::none,
      /* sink = */ sinkMap,
      /* srcs = */ ArrayRef{srcMap},
      /* cons = */ IntegerSet::getEmptySet(0, 0, context),
      /* name = */ "reshape");
}

struct ExpandReshapePass : public ExpandReshapeBase<ExpandReshapePass> {
  void runOnFunction() final;

  // Expand in-order reshape operation
  void expandReshape(ReshapeOp reshapeOp);
};

// Expand in-order reshape operation
void ExpandReshapePass::expandReshape(ReshapeOp reshapeOp) {
  auto src = reshapeOp.tensor();
  auto dst = reshapeOp.getResult();
  auto dstShape = dst.getType().cast<RankedTensorType>().getShape();

  OpBuilder builder(reshapeOp);
  builder.setInsertionPoint(reshapeOp);
  // Try to expand reshape
  Value res = reshapeTensor(builder, src, dstShape);
  if (!res) {
    // If failed, flatten src to a linear buffer
    Value buf = flattenTensor(builder, src);
    // Expand reshape again
    res = reshapeTensor(builder, buf, dstShape);
  }
  assert(res);
  dst.replaceAllUsesWith(res);
  reshapeOp.erase();
}

void ExpandReshapePass::runOnFunction() {
  auto func = getFunction();
  for (auto op : func.getOps<ReshapeOp>()) {
    expandReshape(op);
  }
  return;
}

std::unique_ptr<Pass> createExpandReshapePass() {
  return std::make_unique<ExpandReshapePass>();
}

} // namespace pmlc::dialect::tile
