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
//  stores precede all loads, and all permutations of `op`'s `BlockArgument`s as
//  tiled indexes. (Strictly speaking, since there may be more block args than
//  tiled indexes, it will iterate over all subsets of block args of size
//  `tiledIdxCount` and all permutations of each subset). For each such
//  permutation, `DoStenciling` will verify that each stride requirement
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
//    A `TileSizeGenerator` for each tileable index, used to generate candidate
//    tile sizes for that index.
//  * `requirements`:
//    Stride requirements
//
//    A `llvm::SmallVector<IdxStrideReqs, 8>` which has an `IdxStrideReqs` for
//    each tileable index. These have a function for each I/O Op indicating
//    which strides are legal for the access of this op by this index. For
//    instance, if the op shouldn't use this index, the function should be
//        return stride == 0;
//
//    To build requirements, decide on an order of the tensors and indexes.
//    Importantly, all stores MUST PRECEDE all loads. These orders are otherwise
//    arbitrary. As an example, consider trying to stencil the matmul
//        C[i, j] += A[i, k] * B[k, j]
//    Since the AffineReductionOp writing C is the only store, we make sure it
//    is first in our tensor order and choose {C, A, B} to order our I/O ops and
//    choose {i, j, k} to order our indexes. Then the first element of
//    `requirements` will be the stride requirements for `i`. Looking at the
//    formula above, we see that both `C` and `A` use `i` in the their access,
//    but neither needs `i` to be stride 1 specifically as neither uses `i` for
//    their stride 1 (final) dimension. So their stride verification functions
//    must validate that the stride is non-zero. For `B`, `i` is not used, and
//    so its function must validate that the stride is exactly zero. The order
//    we chose above was C then A then B, so the first element of `requirements`
//    for this example will be
//        IdxStrideReqs{
//          [](int64_t stride) { return stride != 0; }, // C
//          [](int64_t stride) { return stride != 0; }, // A
//          [](int64_t stride) { return stride == 0; }, // B
//        }
//    A similar technique is used to construct the second IdxStrideReqs (for j)
//    and the final IdxStrideReqs (for k).
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
//    otherwise contains the IO ops, with store or reduce ops in `stores` and
//    load ops in `loads`. The order of `loads` and `stores` does not matter, as
//    `DoStencil` will attempt all permutations.
//  * `getCost`:
//    Determine the cost of a proposed tiling. The tiling is provided as
//    parameters to `getCost` (same as for `transform`):
//     * `perm`: A `TensorAndIndexPermutation` which gives the IO ops and the
//       indexes in the same order as `requirements` uses. In particular, this
//       means all store ops will precede all load ops.
//     * `tileSizes`: An `ArrayRef<int64_t>` which gives the size of each index
//       in the selected tiling. Uses the same order of indexes as in `perm` and
//       `requirements`.
//    Returns the cost as a double. If the proposed tiling is illegal, the cost
//    `std::numeric_limits<double>::infinity()` should be returned.
//  * `transform`:
//    Transform `op` based on the already-determined optimal tiling. The tiling
//    is provided as paramters to `transform` (same as for `getCost`):
//     * `perm`: A `TensorAndIndexPermutation` which gives the IO ops and the
//       indexes in the same order `requirements` uses. In particular, this
//       means all store ops will precede all load ops.
//     * `tileSizes`: An `ArrayRef<int64_t>` which gives the size of each index
//       in the selected tiling. Uses the same order of indexes as in `perm` and
//       `requirements`.
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

// For an index, verifiers for each tensor that the index's strides match it
using IdxStrideReqs = llvm::SmallVector<std::function<bool(int64_t)>, 3>;

struct TensorAndIndexPermutation {
  // An order of the Tensors and Indexes used in an operation
  // Note: Tensors are tracked by their load/store/reduce ops, not values,
  // because we need to get stride information, which means we need the op,
  // and for stores/reduces getting to the op from the value is nontrivial
  llvm::SmallVector<mlir::Operation *, 3> ioOps;
  llvm::SmallVector<mlir::BlockArgument, 8> indexes;

  TensorAndIndexPermutation() = default;

  TensorAndIndexPermutation(llvm::ArrayRef<mlir::Operation *> ioOps,
                            llvm::ArrayRef<mlir::BlockArgument> indexes)
      : ioOps(ioOps.begin(), ioOps.end()),
        indexes(indexes.begin(), indexes.end()) {}
};

struct LoadStoreOps {
  // The load and store ops of an AffineParallel
  // Loads and stores are expected to be distinguished within a single op, so
  // are stored separately. Stores and Reduces are not expected to be
  // distinguished within a single op (`capture` may only allow one or the other
  // (or both), but may not distinguish between store and reduce within a single
  // op). Thus, `stores` might have either store or reduce ops.
  llvm::SmallVector<mlir::Operation *, 1> stores;
  llvm::SmallVector<mlir::Operation *, 2> loads;
};

using TileSizeGenerator = std::function<std::vector<int64_t>(int64_t)>;

class StencilBase {
public:
  explicit StencilBase(mlir::AffineParallelOp op, unsigned tiledIdxCount,
                       llvm::ArrayRef<TileSizeGenerator> tilingGenerators,
                       llvm::ArrayRef<IdxStrideReqs> requirements)
      : op(op), tiledIdxCount(tiledIdxCount),
        tilingGenerators(tilingGenerators.begin(), tilingGenerators.end()),
        requirements(requirements.begin(), requirements.end()),
        bestCost(std::numeric_limits<double>::infinity()) {
    assert(tilingGenerators.size() == tiledIdxCount &&
           "Stencil pass requires one tiling generator per tiled index");
    assert(requirements.size() == tiledIdxCount &&
           "Stencil pass requires one requirements vector per tiled index");
    for (auto blockArg : op.getBody()->getArguments()) {
      blockArgs.insert(blockArg);
    }
  }

  // Main function
  void DoStenciling();

protected:
  // Determine if `op` is eligible for stenciling and capture the IO ops if so
  virtual llvm::Optional<LoadStoreOps> capture() = 0;
  // Determine the cost of the specified stencil
  virtual double getCost(TensorAndIndexPermutation perm,
                         ArrayRef<int64_t> tileSize) = 0;
  // Rewrite `op` by applying the specified stencil
  virtual void transform(TensorAndIndexPermutation perm,
                         ArrayRef<int64_t> tileSize) = 0;

  // Get the range of the `idx`th BlockArg
  int64_t getIdxRange(mlir::BlockArgument idx);

  // Call `computeStrideInfo` with caching and automatic conversion to whichever
  // of AffineLoadOp, AffineStoreOp, or AffineReduceOp is correct
  mlir::Optional<mlir::StrideInfo> getStrideInfo(mlir::Operation *ioOp);

  // Print a log of the best stencil (reporting on cost, permutation, and
  // tiling) provided verbosity is at least `logLevel`
  void reportBestStencil(unsigned logLevel);

  // The number of indexes whose semantics must be considered in the tiling
  unsigned getTiledIdxCount() const { return tiledIdxCount; }

  // The BlockArguments of `op` (stored as a set for easy lookup)
  BlockArgumentSet getBlockArgsAsSet() const { return blockArgs; }

  // The ParallelOp that is being stenciled.
  mlir::AffineParallelOp op;

private:
  void BindIndexes(llvm::ArrayRef<mlir::Operation *> ioOps);
  void RecursiveBindIndex(llvm::SmallVector<mlir::BlockArgument, 8> &bound_idxs,
                          llvm::ArrayRef<mlir::Operation *> ioOps);
  void RecursiveTileIndex(const TensorAndIndexPermutation &perm,
                          llvm::MutableArrayRef<int64_t> tileSize,
                          int64_t currIdx);

  // Cached call of the `idx`th tilingGenerator on parameter `range`
  std::vector<int64_t> generateTilings(int64_t idx, int64_t range);

  unsigned tiledIdxCount;
  BlockArgumentSet blockArgs;

  // Cache of results of tilingGenerators calls: First value of the key is which
  // generator was called, second value of the key is range it was called with
  llvm::DenseMap<std::pair<int64_t, int64_t>, std::vector<int64_t>>
      tilingsCache;

  // The range of each index (cached result of op.getConstantRanges())
  llvm::SmallVector<int64_t, 8> ranges;

  // Cache of StrideInfo results
  llvm::DenseMap<mlir::Operation *, mlir::Optional<mlir::StrideInfo>>
      strideInfoCache;

  // The load and store ops
  LoadStoreOps loadsAndStores;

  // For each tiled index, a generator for tile sizes. Ordered to match the
  // index permutation.
  llvm::SmallVector<TileSizeGenerator, 5> tilingGenerators;

  // For each index, a IdxStrideReqs, which provides the functions needed to
  // verify that each store or load op uses this index with the appropriate
  // striding (e.g., stride 0 if the index is unused for that store/load). See
  // the `requirements` section of the documentation at the top of this file for
  // more details
  llvm::SmallVector<IdxStrideReqs, 8> requirements;

  // Note: The bestCost, bestPermutation, and bestTiling all must refer to the
  // same permutation & tiling choices and should only be modified together
  double bestCost;
  TensorAndIndexPermutation bestPermutation;
  llvm::SmallVector<int64_t, 8> bestTiling;
};

} // namespace pmlc::dialect::pxa
