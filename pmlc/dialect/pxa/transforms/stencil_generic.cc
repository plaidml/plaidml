// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/stencil_generic.h"

// TODO: Just seeing if the stencil.cc includes work
#include "mlir/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/tile/ir/ops.h"

#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

// // TODO: includes etc
// #include "mlir/Dialect/Affine/IR/AffineOps.h"

// #include "pmlc/util/logging.h"

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

void StencilGeneric::BindIndexes(
    const llvm::SmallVector<mlir::Value, 3> &tensors) {
  llvm::SmallVector<mlir::BlockArgument, 8> emptyBoundIdxsVector;
  RecursiveBindIndex(&emptyBoundIdxsVector, tensors);
}

// TODO: Better to maintain boundIdxs as both set & vector, or just vector?
//   Or could maintain as map from BlockArg to numeric index (i.e. where it
//   would have been were it a vector)
// TODO: Also, should probably at least be a small vector (how small?)
void StencilGeneric::RecursiveBindIndex(
    llvm::SmallVector<mlir::BlockArgument, 8> *boundIdxs,
    const llvm::SmallVector<mlir::Value, 3> &tensors) {
  auto currIdx = boundIdxs->size();
  if (currIdx == semanticIdxCount) {
    // This is a legal binding, go find a tiling for it
    llvm::SmallVector<int64_t, 8> currTileSize(semanticIdxCount);
    RecursiveTileIndex(TensorAndIndexPermutation(tensors, *boundIdxs),
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
      for (unsigned i = 0; i < tensors.size(); i++) {
        try { // TODO: probably don't keep long term
          if (!requirements.at(std::make_pair(i, currIdx))(tensors[i],
                                                           blockArg)) {
            reqsMet = false;
            break;
          }
        } catch (const std::out_of_range &e) {
          IVLOG(1, "Error message: " << e.what());
          IVLOG(1, "Requested key was: " << std::make_pair(i, currIdx));
          throw;
        }
      }
      if (!reqsMet) {
        continue;
      }

      // If we made it to here, this index has appropriate semantics; bind it
      // and recurse
      boundIdxs->push_back(blockArg);
      RecursiveBindIndex(boundIdxs, tensors);
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
    if (cost < bestCost) {
      bestCost = cost;
      bestPermutation = perm;
      bestTiling = llvm::SmallVector<int64_t, 8>(tileSize->size());
      for (auto sz : *tileSize) {
        bestTiling[sz] = (*tileSize)[sz];
      }
    }
  } else {
    // TODO: Setup cache for the generator
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
    ranges = maybeRanges.getValue(); // TODO: Is this how to use Optional?
  } else {
    IVLOG(4, "Cannot Stencil: Requires constant ranges");
    return;
  }

  auto maybeLoadsAndStores = capture();
  if (maybeLoadsAndStores) {
    loadsAndStores = maybeLoadsAndStores.getValue();
  } else {
    IVLOG(4, "Cannot Stencil: Operations fail to pattern-match.");
    return;
  }

  llvm::SmallVector<Orderer<mlir::Value>, 3> orderableTensors;
  unsigned ord = 0;
  for (auto &loadOp : loadsAndStores.loads) {
    orderableTensors.push_back(Orderer<mlir::Value>(ord++, loadOp.getMemRef()));
  }
  size_t firstStoreIdx = orderableTensors.size();
  for (auto &storeOp : loadsAndStores.stores) {
    orderableTensors.push_back(Orderer<mlir::Value>(ord++, storeOp.out()));
    // TODO: Probably should handle reduces vs. true stores in a different way
    // if (auto reduce_op = llvm::dyn_cast_or_null<AffineReduceOp>(storeOp)) {
    //   tensors.push_back(reduce_op.out());
    // } else if (auto trueStoreOp =
    // llvm::dyn_cast_or_null<mlir::AffineStoreOp>(storeOp)) {
    //   tensors.push_back(trueStoreOp.getMemRef());
    // } else {
    //   // TODO: throw?
    //   IVLOG(1, "Unexpected failure to load tensors from ops in stenciling");
    //   return;
    // }
  }
  auto lastLoadFirstStoreIt = orderableTensors.begin() + firstStoreIdx;
  std::sort(orderableTensors.begin(), lastLoadFirstStoreIt);
  do { // Each load tensor permutation
    std::sort(lastLoadFirstStoreIt, orderableTensors.end());
    do { // Each store tensor permutation
      // Add all legal permutations to legalPermutations
      llvm::SmallVector<mlir::Value, 3> tensors;
      for (const auto &tn : orderableTensors) {
        tensors.push_back(*tn);
      }
      BindIndexes(tensors);
    } while (
        std::next_permutation(lastLoadFirstStoreIt, orderableTensors.end()));
  } while (
      std::next_permutation(orderableTensors.begin(), lastLoadFirstStoreIt));

  if (bestCost < std::numeric_limits<double>::infinity()) {
    transform(bestPermutation, bestTiling);
  } else {
    IVLOG(3, "No legal tiling found to stencil");
  }
}

} // namespace pmlc::dialect::pxa
