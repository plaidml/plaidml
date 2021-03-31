// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/stencil.h"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

namespace {

// A simple wrapper to provide an ordering to object vectors that
// we're going to be processing with std::next_permutation() --
// e.g. if we used pointers as comparison values, our order of
// iteration could vary run-to-run, creating non-determinism.
template <typename V>
class Orderer {
public:
  Orderer(unsigned ord, V value) : ord_{ord}, value_{std::forward<V>(value)} {}

  void setOrd(unsigned ord) { ord_ = ord; }
  unsigned ord() const { return ord_; }

  V &operator*() { return value_; }
  const V &operator*() const { return value_; }

  V &operator->() { return value_; }
  const V &operator->() const { return value_; }

  bool operator<(const Orderer<V> &other) const { return ord() < other.ord(); }

private:
  unsigned ord_;
  V value_;
};

template <typename V>
std::ostream &operator<<(std::ostream &os, const Orderer<V> &v) {
  os << *v << ":" << v.ord();
  return os;
}

template <typename V>
void swap(Orderer<V> &v1, Orderer<V> &v2) {
  unsigned v1o = v1.ord();
  v1.setOrd(v2.ord());
  v2.setOrd(v1o);
  std::swap(*v1, *v2);
}

} // namespace

void StencilBase::reportBestStencil(unsigned logLevel) {
  if (VLOG_IS_ON(logLevel)) {
    std::stringstream bestReport;
    bestReport << "Stencil Selection Report:\n";
    bestReport << "    Best Perf: " << bestCost << "\n";
    std::stringstream tensorPermStr;
    tensorPermStr << "[\n";
    for (auto value : bestPermutation.values) {
      tensorPermStr << "        " << debugString(value) << "\n";
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

int64_t StencilBase::getIdxRange(BlockArgument idx) {
  assert(getBlockArgsAsSet().count(idx) &&
         "getIdxRange only valid on indexes of current op");
  assert(idx.getArgNumber() < ranges.size());
  return ranges[idx.getArgNumber()];
}

Optional<StrideInfo> StencilBase::getStrideInfo(Value value) {
  auto cached = strideInfoCache.find(value);
  if (cached != strideInfoCache.end()) {
    return cached->second;
  }
  auto maybeInfo =
      llvm::TypeSwitch<Operation *, Optional<StrideInfo>>(value.getDefiningOp())
          .Case<PxaLoadOp>([&](PxaLoadOp op) { return computeStrideInfo(op); })
          .Case<PxaReduceOp>(
              [&](PxaReduceOp op) { return computeStrideInfo(op); })
          .Default([](Operation *) { return llvm::None; });
  strideInfoCache[value] = maybeInfo;
  return maybeInfo;
}

void StencilBase::BindIndexes(llvm::ArrayRef<Value> values) {
  llvm::SmallVector<BlockArgument, 8> emptyBoundIdxsVector;
  RecursiveBindIndex(emptyBoundIdxsVector, values);
}

void StencilBase::RecursiveBindIndex(
    llvm::SmallVector<BlockArgument, 8> &boundIdxs,
    llvm::ArrayRef<Value> values) {
  auto currIdx = boundIdxs.size();
  if (currIdx == tiledIdxCount) {
    // This is a legal binding, go find a tiling for it
    llvm::SmallVector<int64_t, 8> currTileSize(tiledIdxCount);
    RecursiveTileIndex(TensorAndIndexPermutation(values, boundIdxs),
                       currTileSize, 0);
  } else {
    for (const auto blockArg : getBlockArgsAsSet()) {
      // Don't bind same index twice
      // Note: While it's awkward to be repeatedly searching a vector, I think
      // boundIdxs is small enough that it would not be efficient to maintain a
      // parallel map
      if (std::find(boundIdxs.begin(), boundIdxs.end(), blockArg) !=
          boundIdxs.end()) {
        continue;
      }

      // Verify the requirements for this index with each tensor are all met
      bool reqsMet = true;
      assert(requirements[currIdx].size() == values.size() &&
             "Each requirements entry must have one function per I/O op");
      for (unsigned i = 0; i < values.size(); i++) {
        auto strideInfo = getStrideInfo(values[i]);
        auto stride = strideInfo->strides[blockArg];
        if (!requirements[currIdx][i](stride)) {
          reqsMet = false;
          break;
        }
      }
      if (!reqsMet) {
        continue;
      }

      // If we made it to here, this index has appropriate semantics; bind it
      // and recurse
      boundIdxs.push_back(blockArg);
      RecursiveBindIndex(boundIdxs, values);
      boundIdxs.pop_back();
    }
  }
}

void StencilBase::RecursiveTileIndex(        //
    const TensorAndIndexPermutation &perm,   //
    llvm::MutableArrayRef<int64_t> tileSize, //
    int64_t currIdx) {
  assert(tileSize.size() == tiledIdxCount);
  if (currIdx == tiledIdxCount) {
    auto cost = getCost(perm, tileSize);
    if (VLOG_IS_ON(3)) {
      std::stringstream currTilingStr;
      currTilingStr << "[ ";
      for (const auto &sz : tileSize) {
        currTilingStr << sz << " ";
      }
      currTilingStr << "]";
      IVLOG(3, "Considering Tiling " << currTilingStr.str()
                                     << ", which would have cost " << cost);
    }
    if (cost < bestCost) {
      bestCost = cost;
      bestPermutation = perm;
      bestTiling.assign(tileSize.begin(), tileSize.end());
    }
  } else {
    assert(getBlockArgsAsSet().count(perm.indexes[currIdx]) &&
           "BlockArg for current index must be valid");
    for (int64_t currIdxTileSize : generateTilings(
             currIdx, ranges[perm.indexes[currIdx].getArgNumber()])) {
      tileSize[currIdx] = currIdxTileSize;
      RecursiveTileIndex(perm, tileSize, currIdx + 1);
    }
  }
}

void StencilBase::DoStenciling() {
  // Initialization
  auto maybeRanges = op.getConstantRanges();
  if (!maybeRanges) {
    IVLOG(4, "Cannot Stencil: Requires constant ranges");
    return;
  }
  ranges = *maybeRanges;
  assert(ranges.size() == getBlockArgsAsSet().size());

  auto maybeLoadsAndStores = capture();
  if (!maybeLoadsAndStores) {
    IVLOG(4, "Cannot Stencil: Operations fail to pattern-match.");
    return;
  }
  loadsAndStores = *maybeLoadsAndStores;

  // We wrap loads & stores with `Orderer` to make the order the permutations
  // are iterated through deterministic (the "sorted" order of the IO ops is the
  // order they were returned by `capture`) -- without this, the sorted order
  // would be however the pointers were ordered in memory.
  llvm::SmallVector<Orderer<Value>, 3> ordered;
  unsigned ord = 0;
  for (auto &storeOp : loadsAndStores.stores) {
    ordered.push_back(Orderer<Value>(ord++, storeOp));
  }
  size_t firstLoadIdx = ordered.size();
  for (auto &loadOp : loadsAndStores.loads) {
    ordered.push_back(Orderer<Value>(ord++, loadOp));
  }
  auto itLastStoreFirstLoad = ordered.begin() + firstLoadIdx;
  std::sort(ordered.begin(), itLastStoreFirstLoad);
  do { // Each store tensor permutation
    std::sort(itLastStoreFirstLoad, ordered.end());
    do { // Each load tensor permutation
      llvm::SmallVector<Value, 3> values;
      for (const auto &ioOp : ordered) {
        values.push_back(*ioOp);
      }
      BindIndexes(values);
    } while (std::next_permutation(itLastStoreFirstLoad, ordered.end()));
  } while (std::next_permutation(ordered.begin(), itLastStoreFirstLoad));

  if (bestCost < std::numeric_limits<double>::infinity()) {
    reportBestStencil(2);
    transform(bestPermutation, bestTiling);
  } else {
    IVLOG(3, "No legal tiling found to stencil");
  }
}

} // namespace pmlc::dialect::pxa
