// Copyright 2020 Intel Corporation

#pragma once

//
// Provides a base class `StencilBase` for building stencil passes.
//
// If you would like to look at an example stencil pass to supplement this
// documentation, one can be found at `pmlc/target/x86/stencil_xsmm.cc`.
//
// To create a derived stencil pass from `StencilBase`, you will need to pass
// appropriate parameters for the `StencilBase` constructor and to overload the
// virtual functions `capture`, `getCost`, and `transform`.
//
// The main function is `DoStenciling`, and is what the pass should call to
// perform the stenciling.
//
// `DoStenciling` Overview
// -----------------------
//  `DoStenciling` will first find appropriate IO ops (i.e., loads
//  (`mlir::AffineLoadOp`) and stores (`mlir::AffineStoreOp` or
//  `pxa::AffineReduceOp`)) using the `capture` function.
//
//  It will then iterate through all permutations of the IO ops that have all
//  loads precede all stores, and all permutations of `op`'s `BlockArgument`s as
//  tiled indexes. (Strictly speaking, since there may be more block args than
//  tiled indexes, it will iterate over all subsets of block args of size
//  `tiledIdxCount` and all permutations of each subset). For each such
//  permutation, `DoStenciling` will verify that each tensor-index requirement
//  specified in `requirements` is met. This logic includes shortcutting to skip
//  iterating through permutations that are already known to fail.
//
//  For tensor & index permutations that meet all requirements, `DoStenciling`
//  will use the `tilingGenerators` to generate potential tile sizes for each
//  index. These will be evaluated using the `getCost` function.
//
//  If any tilings with finite cost have been generated, `DoStenciling` will use
//  whichever is cheapest in the `transform` function to rewrite `op`.
//
// Constructor Parameters
// ----------------------
//  * `op`:
//    This `mlir::AffineParallelOp` is the op that the current instance will
//    stencil.
//  * `tiledIdxCount`:
//    How many indexes will be considered for tiling.
//  * `tilingGenerators`:
//    A `TileSizeGenerator` for each tileable index, used to generate proposed
//    tile sizes for that index.
//  * `requirements`:
//    Pairwise tensor-index requirements.
//
//    Maps integer pairs representing IO op order and index order to a function
//    that determines if a given IO op and BlockArg pair are valid if used in
//    that order. As an example, consider the trying to match the matmul
//        C[i, j] += A[i, k] * B[k, j]
//    To build requirements, decide on an order of the tensors and indexes
//    (these orders are arbitrary except that all stores must follow all loads).
//    Here let's choose {A, B, C} and {i, j, k} as our order. Then to express
//    the requirement "`j` must be a stride one index of `C`", for the key
//    (2, 1) we set the value to be a function that returns `true` if and only
//    if the passed BlockArg is a stride one index of the passed Operation*.
//
//  The `op` parameter will be different for each instance of the pass -- it is
//  the operation that MLIR is trying to stencil. The other constructor
//  parameters will commonly be fixed amongst all instances of a derived class,
//  although they can be configurable if that is useful to the derived class.
//
// Virtual Functions to Overload
// -----------------------------
//  * `capture`:
//    Search the body of `op` for IO ops, and verify that `op` has a structure
//    amenable to this stenciling. Returns a `llvm::Optional<LoadStoreOps>`,
//    which is to be `None` if `op` cannot be stenciled by this pass and
//    otherwise contains the IO ops, with load ops in `loads` and store or
//    reduce ops in `stores`. The order of `loads` and `stores` does not matter,
//    as `DoStencil` will attempt all permutations.
//  * `getCost`:
//    Determine the cost of a proposed tiling. The tiling is provided as
//    parameters to `getCost` (same as for `transform`):
//     * `perm`: A `TensorAndIndexPermutation` which gives the IO ops and the
//       indexes in semantic order.
//     * `tileSizes`: An `ArrayRef<int64_t>` which gives the size of each index
//       in the selected tiling. Uses the same order of indexes as in `perm`.
//    Returns the cost as a double. If the proposed tiling is illegal, the cost
//    `std::numeric_limits<double>::infinity()` should be returned.
//  * `transform`:
//    Transform `op` based on the already-determined optimal tiling. The tiling
//    is provided as paramters to `transform` (same as for `getCost`):
//     * `perm`: A `TensorAndIndexPermutation` which gives the IO ops and the
//       indexes in semantic order.
//     * `tileSizes`: An `ArrayRef<int64_t>` which gives the size of each index
//       in the selected tiling. Uses the same order of indexes as in `perm`.
//    The `transform` function will also need to access the member variable
//    `op`, as this is the operation it is transforming.
//

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/pxa/analysis/strides.h"

namespace pmlc::dialect::pxa {

using BlockArgumentSet = llvm::SmallPtrSet<mlir::BlockArgument, 8>;

struct TensorAndIndexPermutation {
  // An order of the Tensors and Indexes used in an operation
  // Note: Tensors are tracked by their load/store/reduce ops, not values,
  // because we need to get stride information, which means we need the op,
  // and for stores/reduces getting to the op from the value is nontrivial
  llvm::SmallVector<mlir::Operation *, 3> ioOps;
  llvm::SmallVector<mlir::BlockArgument, 8> indexes;

  TensorAndIndexPermutation() = default;

  TensorAndIndexPermutation(llvm::SmallVector<mlir::Operation *, 3> ioOps,
                            llvm::SmallVector<mlir::BlockArgument, 8> indexes)
      : ioOps(ioOps), indexes(indexes) {}
};

struct LoadStoreOps {
  // The load and store ops of an AffineParallel
  // Loads and stores are expected to be distinguished within a single op, so
  // are stored separately. Stores and Reduces are not expected to be
  // distinguished within a single op (`capture` may only allow one or the other
  // (or both), but almost no cases will have both a store and a reduce within a
  // single op). Thus, `stores` might have either store or reduce ops.
  llvm::SmallVector<mlir::Operation *, 2> loads;
  llvm::SmallVector<mlir::Operation *, 1> stores;
};

using TileSizeGenerator = std::function<std::vector<int64_t>(int64_t)>;

class StencilBase {
private:
  void BindIndexes(const llvm::SmallVector<mlir::Operation *, 3> &ioOps);
  void RecursiveBindIndex(llvm::SmallVector<mlir::BlockArgument, 8> *bound_idxs,
                          const llvm::SmallVector<mlir::Operation *, 3> &ioOps);
  void RecursiveTileIndex(const TensorAndIndexPermutation &perm,
                          llvm::SmallVector<int64_t, 8> *tileSize,
                          int64_t currIdx);

protected: // TODO: private backend for some of this?
  virtual llvm::Optional<LoadStoreOps> capture() = 0;
  virtual double getCost(TensorAndIndexPermutation perm,
                         ArrayRef<int64_t> tileSize) = 0;
  virtual void transform(TensorAndIndexPermutation perm,
                         ArrayRef<int64_t> tileSize) = 0;
  int64_t getIdxRange(mlir::BlockArgument idx);
  mlir::Optional<mlir::StrideInfo> getStrideInfo(mlir::Operation *ioOp);
  void reportBestStencil(unsigned logLevel);

  // Cache of StrideInfo results
  llvm::DenseMap<mlir::Operation *, mlir::StrideInfo> strideInfoCache;

  // The ParallelOp that is being stenciled.
  mlir::AffineParallelOp op;

  // The BlockArguments of `op` (stored as a set for easy lookup)
  BlockArgumentSet blockArgs;

  // The load and store ops
  LoadStoreOps loadsAndStores;

  // The range of each index (cached result of op.getConstantRanges())
  llvm::SmallVector<int64_t, 8> ranges;

  // The number of indexes whose semantics must be considered in the tiling
  unsigned tiledIdxCount;

  // For each tiled index, a generator for tile sizes. Ordered to match the
  // index permutation.
  llvm::SmallVector<TileSizeGenerator, 5> tilingGenerators;

  // For each tensor/index semantic pair (given as a pair of `int64_t`s), a
  // function to determine if the load or store op of a tensor and the BlockArg
  // of an index meet the requirements of that pair.
  llvm::DenseMap<std::pair<int64_t, int64_t>,
                 std::function<bool(mlir::Operation *, mlir::BlockArgument)>>
      requirements;

  double bestCost;
  TensorAndIndexPermutation bestPermutation;
  llvm::SmallVector<int64_t, 8>
      bestTiling; // only makes sense paired with `bestPermutation`

public:
  explicit StencilBase(
      mlir::AffineParallelOp op, unsigned tiledIdxCount,
      llvm::SmallVector<TileSizeGenerator, 5> tilingGenerators,
      llvm::DenseMap<
          std::pair<int64_t, int64_t>,
          std::function<bool(mlir::Operation *, mlir::BlockArgument)>>
          requirements)
      : op(op), tiledIdxCount(tiledIdxCount),
        tilingGenerators(tilingGenerators), requirements(requirements),
        bestCost(std::numeric_limits<double>::infinity()) {
    assert(tilingGenerators.size() == tiledIdxCount &&
           "Stencil pass requires one tiling generator per tiled index");
    for (const auto &kvp : requirements) {
      assert(kvp.first.second >= 0 &&
             "Only nonnegative indexes are valid in requirements");
      assert(kvp.first.second < tiledIdxCount &&
             "Only tiled indexes are valid in requirements");
    }
    for (auto blockArg : op.getBody()->getArguments()) {
      blockArgs.insert(blockArg);
    }
  }

  // Main function
  void DoStenciling();
};

} // namespace pmlc::dialect::pxa
