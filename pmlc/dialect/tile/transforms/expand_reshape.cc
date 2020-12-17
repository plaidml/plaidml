// Copyright 2020, Intel Corporation

#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/pass_detail.h"

#include <numeric>
#include <iostream>

using namespace mlir;                // NOLINT
using namespace pmlc::dialect::stdx; // NOLINT

namespace pmlc::dialect::tile {

namespace {

struct ExpandReshapePass : public ExpandReshapeBase<ExpandReshapePass> {
  void runOnFunction() final;

  void expandReshape(ReshapeOp reshapeOp);
};

void ExpandReshapePass::expandReshape(ReshapeOp reshapeOp) {
  auto src = reshapeOp.tensor();
  auto srcShape = src.getType().cast<RankedTensorType>().getShape();
  SmallVector<int64_t, 4> srcDims(srcShape.begin(), srcShape.end());
  auto dest = reshapeOp.getResult();
  auto destShape = dest.getType().cast<RankedTensorType>().getShape();
  SmallVector<int64_t, 4> destDims(destShape.begin(), destShape.end());

  // Dump
  for (unsigned dim : srcDims) std::cerr << dim << " ";
  std::cerr << "\n";
  for (unsigned dim : destDims) std::cerr << dim << " ";
  std::cerr << "\n";

  SmallVector<int64_t, 4> idxs;
  SmallVector<SmallVector<unsigned, 4>, 4> srcIdxs(srcDims.size());
  SmallVector<SmallVector<unsigned, 4>, 4> destIdxs(destDims.size());

  int pSrc = srcDims.size() - 1;
  int pDest = destDims.size() - 1;
  while (pSrc >= 0 && pDest >=0) {
    while (pSrc >= 0 && srcDims[pSrc] == 1) {
      --pSrc;
    }
    if (pSrc < 0) {
      break;
    }
    while (pDest >= 0 && destDims[pDest] == 1) {
      --pDest;
    }
    if (pDest < 0) {
      break;
    }
    int64_t gcd = std::gcd(srcDims[pSrc], destDims[pDest]);
    unsigned idxId = idxs.size();
    idxs.emplace_back(gcd);
    srcIdxs[pSrc].emplace_back(idxId);
    destIdxs[pDest].emplace_back(idxId);
    srcDims[pSrc] /= gcd;
    destDims[pDest] /= gcd;
  } 

  while (pSrc >= 0) {
    if (srcDims[pSrc] > 1) {
      unsigned idxId = idxs.size();
      idxs.emplace_back(srcDims[pSrc]);
      srcIdxs[pSrc].emplace_back(idxId);
    }
    --pSrc;
  }
  while (pDest >= 0) {
    if (destDims[pDest] > 1) {
      unsigned idxId = idxs.size();
      idxs.emplace_back(destDims[pDest]);
      destIdxs[pDest].emplace_back(idxId);
    }
    --pDest;
  }

  // Dump
  std::cerr << "Idxs: ";
  for (auto idx : idxs) std::cerr << idx << " ";
  std::cerr << "\n";
  for (unsigned i = 0; i < srcDims.size(); ++i) {
    std::cerr << i << ": ";
    auto &ids = srcIdxs[i];
    for (auto j : ids) std::cerr << j << " ";
    std::cerr << "\n";
  }
  for (unsigned i = 0; i < destDims.size(); ++i) {
    std::cerr << i << ": ";
    auto &ids = destIdxs[i];
    for (auto j : ids) std::cerr << j << " ";
    std::cerr << "\n";
  }
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
