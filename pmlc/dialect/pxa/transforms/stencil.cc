// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/stencil.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/util/logging.h"

namespace pmlc::dialect::pxa {

void StencilBase::reportBestStencil(unsigned logLevel) {
  if (VLOG_IS_ON(logLevel)) {
    std::stringstream bestReport;
    bestReport << "Stencil Selection Report:\n";
    bestReport << "    Best Perf: " << bestCost << "\n";
    std::stringstream tensorPermStr;
    tensorPermStr << "[\n";
    for (auto ioOp : bestPermutation.ioOps) {
      tensorPermStr << "        " << mlir::debugString(*ioOp) << "\n";
    }
    tensorPermStr << "    ]";
    bestReport << "    Best Tensor Permutation: " << tensorPermStr.str()
               << "\n";
    std::stringstream indexPermStr;
    indexPermStr << "[ ";
    for (auto ind : bestPermutation.indexes) {
      assert(getBlockArgsAsSet().count(ind) &&
             "All tiled indexes must be introduced in current loop");
      indexPermStr << ind.getArgNumber() << " ";
    }
    indexPermStr << "]";
    bestReport << "    Best Index Permutation: " << indexPermStr.str() << "\n";
    std::stringstream bestTilingStr;
    bestTilingStr << "[ ";
    for (const auto &sz : bestTiling) {
      bestTilingStr << sz << " ";
    }
    bestTilingStr << "]";
    bestReport << "    Best Tiling: " << bestTilingStr.str();
    IVLOG(logLevel, bestReport.str());
  }
}

std::vector<int64_t> StencilBase::generateTilings(int64_t idx, int64_t range) {
  std::pair<int64_t, int64_t> idxRangePair(idx, range);
  auto cached = tilingsCache.find(idxRangePair);
  if (cached != tilingsCache.end()) {
    return cached->second;
  }
  auto result = tilingGenerators[idx](range);
  tilingsCache.insert(std::make_pair(idxRangePair, result));
  return result;
}

int64_t StencilBase::getIdxRange(mlir::BlockArgument idx) {
  assert(getBlockArgsAsSet().count(idx) &&
         "getIdxRange only valid on indexes of current op");
  assert(idx.getArgNumber() < ranges.size());
  return ranges[idx.getArgNumber()];
}

mlir::Optional<mlir::StrideInfo>
StencilBase::getStrideInfo(mlir::Operation *op) {
  // TODO: As written, this doesn't cache `None` returns; I think that's what we
  // want?
  auto cached = strideInfoCache.find(op);
  if (cached != strideInfoCache.end()) {
    return cached->second;
  }
  auto loadOp = llvm::dyn_cast<mlir::AffineLoadOp>(*op);
  if (loadOp) {
    auto strideInfo = computeStrideInfo(loadOp);
    if (strideInfo.hasValue())
      strideInfoCache.insert(std::make_pair(op, strideInfo.getValue()));
    return strideInfo;
  }
  auto storeOp = llvm::dyn_cast<mlir::AffineStoreOp>(*op);
  if (storeOp) {
    auto strideInfo = computeStrideInfo(storeOp);
    if (strideInfo.hasValue())
      strideInfoCache.insert(std::make_pair(op, strideInfo.getValue()));
    return strideInfo;
  }
  auto reduceOp = llvm::dyn_cast<AffineReduceOp>(*op);
  if (reduceOp) {
    auto strideInfo = computeStrideInfo(reduceOp);
    if (strideInfo.hasValue())
      strideInfoCache.insert(std::make_pair(op, strideInfo.getValue()));
    return strideInfo;
  }
  return llvm::None;
}

void StencilBase::BindIndexes(
    const llvm::SmallVector<mlir::Operation *, 3> &ioOps) {
  llvm::SmallVector<mlir::BlockArgument, 8> emptyBoundIdxsVector;
  RecursiveBindIndex(&emptyBoundIdxsVector, ioOps);
}

void StencilBase::RecursiveBindIndex(
    llvm::SmallVector<mlir::BlockArgument, 8> *boundIdxs,
    const llvm::SmallVector<mlir::Operation *, 3> &ioOps) {
  auto currIdx = boundIdxs->size();
  if (currIdx == tiledIdxCount) {
    // This is a legal binding, go find a tiling for it
    llvm::SmallVector<int64_t, 8> currTileSize(tiledIdxCount);
    RecursiveTileIndex(TensorAndIndexPermutation(ioOps, *boundIdxs),
                       &currTileSize, 0);
  } else {
    for (const auto &blockArg : getBlockArgsAsSet()) {
      // Don't bind same index twice
      // Note: While it's awkward to be repeatedly searching a vector, I think
      // boundIdxs is small enough that it would not be efficient to maintain a
      // parallel map
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

void StencilBase::RecursiveTileIndex(        //
    const TensorAndIndexPermutation &perm,   //
    llvm::SmallVector<int64_t, 8> *tileSize, //
    int64_t currIdx) {
  assert(tileSize->size() == tiledIdxCount);
  if (currIdx == tiledIdxCount) {
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
    assert(getBlockArgsAsSet().count(perm.indexes[currIdx]) &&
           "BlockArg for current index must be valid");
    for (int64_t currIdxTileSize : generateTilings(
             currIdx, ranges[perm.indexes[currIdx].getArgNumber()])) {
      (*tileSize)[currIdx] = currIdxTileSize;
      RecursiveTileIndex(perm, tileSize, currIdx + 1);
    }
  }
}

void StencilBase::DoStenciling() {
  // Initialization
  auto maybeRanges = op.getConstantRanges();
  if (maybeRanges) {
    ranges = *maybeRanges;
  } else {
    IVLOG(4, "Cannot Stencil: Requires constant ranges");
    return;
  }
  assert(ranges.size() == getBlockArgsAsSet().size());

  auto maybeLoadsAndStores = capture();
  if (maybeLoadsAndStores) {
    loadsAndStores = *maybeLoadsAndStores;
  } else {
    IVLOG(4, "Cannot Stencil: Operations fail to pattern-match.");
    return;
  }

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
    reportBestStencil(2);
    transform(bestPermutation, bestTiling);
  } else {
    IVLOG(3, "No legal tiling found to stencil");
  }
}

} // namespace pmlc::dialect::pxa
