// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/stencil_generic.h"

// TODO: Just seeing if the stencil.cc includes work
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/passes.h"

#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

// // TODO: includes etc
// #include "mlir/Dialect/Affine/IR/AffineOps.h"

// #include "pmlc/util/logging.h"

namespace pmlc::dialect::pxa {

int64_t StencilGeneric::getIdxRange(mlir::BlockArgument idx) {
  assert(blockArgs.count(idx) && "getIdxRange should have block arg in block");
  assert(idx.getArgNumber() < ranges.size());
  assert(idx.getArgNumber() >= 0 && "TODO scrap");
  assert(idx.getArgNumber() <= 2 && "TODO scrap");
  IVLOG(5, "inside getIdxRange");
  IVLOG(4, "arg number: " << idx.getArgNumber()); // Intermittent crashes here
  IVLOG(4, "ranges: ");
  for (auto r : ranges) {
    IVLOG(4, "  " << r);
  }
  IVLOG(4, "requested range: " << ranges[idx.getArgNumber()]);
  return ranges[idx.getArgNumber()];
}

mlir::Optional<mlir::StrideInfo>
StencilGeneric::getStrideInfo(mlir::Operation *op) {
  auto cached = strideInfoCache.find(op);
  if (cached != strideInfoCache.end()) {
    return cached->second;
  }
  auto loadOp = llvm::dyn_cast<mlir::AffineLoadOp>(*op);
  if (loadOp) {
    auto strideInfo = computeStrideInfo(loadOp);
    if (strideInfo.hasValue())
      strideInfoCache.emplace(std::make_pair(op, strideInfo.getValue()));
    return strideInfo;
  }
  auto storeOp = llvm::dyn_cast<mlir::AffineStoreOp>(*op);
  if (storeOp) {
    auto strideInfo = computeStrideInfo(storeOp);
    if (strideInfo.hasValue())
      strideInfoCache.emplace(std::make_pair(op, strideInfo.getValue()));
    return strideInfo;
  }
  auto reduceOp = llvm::dyn_cast<AffineReduceOp>(*op);
  if (reduceOp) {
    auto strideInfo = computeStrideInfo(reduceOp);
    if (strideInfo.hasValue())
      strideInfoCache.emplace(std::make_pair(op, strideInfo.getValue()));
    return strideInfo;
  }
  return llvm::None;
}

void StencilGeneric::BindIndexes(
    const llvm::SmallVector<mlir::Operation *, 3> &ioOps) {
  llvm::SmallVector<mlir::BlockArgument, 8> emptyBoundIdxsVector;
  RecursiveBindIndex(&emptyBoundIdxsVector, ioOps);
}

// TODO: Better to maintain boundIdxs as both set & vector, or just vector?
//   Or could maintain as map from BlockArg to numeric index (i.e. where it
//   would have been were it a vector)
// TODO: Also, should probably at least be a small vector (how small?)
void StencilGeneric::RecursiveBindIndex(
    llvm::SmallVector<mlir::BlockArgument, 8> *boundIdxs,
    const llvm::SmallVector<mlir::Operation *, 3> &ioOps) {
  auto currIdx = boundIdxs->size();
  if (currIdx == semanticIdxCount) {
    // This is a legal binding, go find a tiling for it
    llvm::SmallVector<int64_t, 8> currTileSize(semanticIdxCount);
    RecursiveTileIndex(TensorAndIndexPermutation(ioOps, *boundIdxs),
                       &currTileSize, 0);
  } else {
    for (const auto &blockArg : blockArgs) {
      // Don't bind same index twice
      if (std::find(boundIdxs->begin(), boundIdxs->end(), blockArg) !=
          boundIdxs->end()) {
        continue;
      }

      // Verify the requirements for this index with each tensor are all met
      bool reqsMet = true;
      for (unsigned i = 0; i < ioOps.size(); i++) {
        auto it = requirements.find(std::make_pair(i, currIdx));
        if (it != requirements.end() && !it->second(ioOps[i], blockArg)) {
          reqsMet = false;
          break;
        }
      }
      if (!reqsMet) {
        continue;
      }

      // If we made it to here, this index has appropriate semantics; bind it
      // and recurse
      boundIdxs->push_back(blockArg);
      RecursiveBindIndex(boundIdxs, ioOps);
      boundIdxs->pop_back();
    }
  }
}

void StencilGeneric::RecursiveTileIndex(     //
    const TensorAndIndexPermutation &perm,   //
    llvm::SmallVector<int64_t, 8> *tileSize, //
    int64_t currIdx) {
  assert(tileSize->size() == semanticIdxCount);
  if (currIdx == semanticIdxCount) {
    auto cost = getCost(perm, *tileSize);
    if (VLOG_IS_ON(3)) {
      std::stringstream currTilingStr;
      currTilingStr << "[ ";
      for (const auto &sz : *tileSize) {
        currTilingStr << sz << " ";
      }
      currTilingStr << "]";
      IVLOG(3, "Considering Tiling " << currTilingStr.str()
                                     << ", which would have cost " << cost);
    }
    if (cost < bestCost) {
      bestCost = cost;
      bestPermutation = perm;
      bestTiling = *tileSize;
    }
  } else {
    // TODO: Setup cache for the generator
    assert(blockArgs.count(perm.indexes[currIdx]) &&
           "BlockArg for current index must be valid");
    for (int64_t currIdxTileSize : tilingGenerators[currIdx](
             ranges[perm.indexes[currIdx].getArgNumber()])) {
      (*tileSize)[currIdx] = currIdxTileSize;
      RecursiveTileIndex(perm, tileSize, currIdx + 1);
    }
  }
}

void StencilGeneric::DoStenciling() {
  // Initialization
  auto maybeRanges = op.getConstantRanges();
  if (maybeRanges) {
    ranges = *maybeRanges;
  } else {
    IVLOG(4, "Cannot Stencil: Requires constant ranges");
    return;
  }

  auto maybeLoadsAndStores = capture();
  if (maybeLoadsAndStores) {
    loadsAndStores = *maybeLoadsAndStores;
  } else {
    IVLOG(4, "Cannot Stencil: Operations fail to pattern-match.");
    return;
  }

  // TODO: Rework this section to mlir::Operation *ioOps
  // TODO: Deal with nondeterminisitic order
  llvm::SmallVector<mlir::Operation *, 3> ioOps;
  for (auto &loadOp : loadsAndStores.loads) {
    ioOps.push_back(loadOp);
  }
  unsigned firstStoreIdx = ioOps.size();
  for (auto &storeOp : loadsAndStores.stores) {
    ioOps.push_back(storeOp);
  }
  auto lastLoadFirstStoreIt = ioOps.begin() + firstStoreIdx;
  std::sort(ioOps.begin(), lastLoadFirstStoreIt);
  do { // Each load tensor permutation
    std::sort(lastLoadFirstStoreIt, ioOps.end());
    do { // Each store tensor permutation
      BindIndexes(ioOps);
    } while (std::next_permutation(lastLoadFirstStoreIt, ioOps.end()));
  } while (std::next_permutation(ioOps.begin(), lastLoadFirstStoreIt));

  if (bestCost < std::numeric_limits<double>::infinity()) {
    transform(bestPermutation, bestTiling);
  } else {
    IVLOG(3, "No legal tiling found to stencil");
  }
}

} // namespace pmlc::dialect::pxa
