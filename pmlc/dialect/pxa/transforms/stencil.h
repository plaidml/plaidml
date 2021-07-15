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
// The main function is `performStenciling`, and is what the pass should call to
// perform the stenciling.
//
// Constructor Parameters
// ----------------------
//  * `op`:
//    This `mlir::AffineParallelOp` is the op that the current instance will
//    stencil.
//  * `requirements`:
//    A vector of `StencilIndexRequirement` elements, one entry for each
//    tileable index.
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
//    for this example will be:
//        StencilIndexRequirement{ // requirement for i
//          /*idxName=*/"i",
//          /*tilingGenerator=*/EvenTilingGenerator(),
//          /*predicates=*/IndexStridePredicates{
//            [](int64_t stride) { return stride != 0; }, // C
//            [](int64_t stride) { return stride != 0; }, // A
//            [](int64_t stride) { return stride == 0; }, // B
//          }},
//    A similar technique is used to construct the other StencilIndexRequirement
//    for j and k.
//

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/util/schedule.h"

namespace pmlc::dialect::pxa {

using BlockArgumentSet = mlir::SmallPtrSet<mlir::BlockArgument, 8>;

using IndexStridePredicate = std::function<bool(int64_t)>;
using IndexStridePredicates = mlir::SmallVector<IndexStridePredicate>;

using TileSizeGenerator = std::function<std::vector<int64_t>(int64_t)>;

struct ValueStrideInfo {
  mlir::Value value;
  StrideInfo strideInfo;
};

struct StencilIndexRequirement {
  std::string idxName;

  // For each tiled index, a generator for tile sizes. Ordered to match the
  // index permutation.
  TileSizeGenerator tilingGenerator;

  // A function for each I/O op indicating which strides are legal for the
  // access of this op by this index. For instance, if the op shouldn't use this
  // index, the function should be:
  //   return stride == 0;
  IndexStridePredicates predicates;

  bool check(mlir::ArrayRef<ValueStrideInfo> values,
             mlir::BlockArgument blockArg) const;
};

// An order of the Tensors and indicies used in an operation
struct StencilOption {
  mlir::SmallVector<ValueStrideInfo, 3> values;
  mlir::SmallVector<mlir::BlockArgument, 8> indexes;

  StencilOption() = default;

  StencilOption(mlir::ArrayRef<ValueStrideInfo> values,
                mlir::ArrayRef<mlir::BlockArgument> indexes)
      : values(values.begin(), values.end()),
        indexes(indexes.begin(), indexes.end()) {}
};

// The load and store values captured within the body of an `affine.parallel`.
struct StencilCapture {
  mlir::SmallVector<mlir::Value, 1> stores;
  mlir::SmallVector<mlir::Value, 2> loads;
};

class StencilBase {
public:
  explicit StencilBase(mlir::AffineParallelOp op,
                       mlir::ArrayRef<StencilIndexRequirement> requirements);

  // `performStenciling` will first find appropriate IO ops (i.e., loads
  // (`PxaLoadOp`) and stores (pxa::PxaReduceOp`)) using the `capture`
  // function.
  //
  // It will then iterate through all permutations of the IO ops that have all
  // stores precede all loads, and all permutations of `op`'s `BlockArgument`s
  // as tiled indexes. (Strictly speaking, since there may be more block args
  // than tiled indexes, it will iterate over all subsets of block args of size
  // `tiledIdxCount` and all permutations of each subset). For each such
  // permutation, `performStenciling` will verify that each stride requirement
  // specified in `requirements` is met. This logic includes shortcutting to
  // skip iterating through permutations that are already known to fail.
  //
  // For tensor & index permutations that meet all requirements,
  // `performStenciling` will use the `tilingGenerators` to generate potential
  // tile sizes for each index. These will be evaluated using the `getCost`
  // function.
  //
  // If any tilings with finite cost have been generated, `performStenciling`
  // will use whichever is cheapest in the `transform` function to rewrite
  // `op`.
  void performStenciling();

protected:
  // Search the body of `op` for IO ops, and verify that `op` has a structure
  // amenable to this stenciling. Returns a `mlir::Optional<StencilCapture>`,
  // which is to be `None` if `op` cannot be stenciled by this pass and
  // otherwise contains the IO ops, with store or reduce ops in `stores` and
  // load ops in `loads`. The order of `loads` and `stores` does not matter,
  // as `performStencil` will attempt all permutations.
  virtual mlir::Optional<StencilCapture> capture() = 0;

  // Determine the cost of a proposed tiling. The tiling is provided as
  // parameters to `getCost` (same as for `transform`):
  //  * `stencil`: A `StencilOption` which gives the IO ops and the
  //    indexes in the same order as `requirements` uses. In particular, this
  //    means all store ops will precede all load ops.
  //  * `tileSizes`: An `ArrayRef<int64_t>` which gives the size of each
  //    index in the selected tiling. Uses the same order of indexes as in
  //    `perm` and `requirements`.
  // Returns the cost as a double. If the proposed tiling is illegal, the
  // cost `std::numeric_limits<double>::infinity()` should be returned.
  virtual double getCost(const StencilOption &stencil,
                         mlir::ArrayRef<int64_t> tileSizes) = 0;

  // Transform `op` based on the already-determined optimal tiling. The
  // tiling is provided as paramters to `transform` (same as for `getCost`):
  //  * `stencil`: A `StencilOption` which gives the IO ops and the
  //    indexes in the same order `requirements` uses. In particular, this
  //    means all store ops will precede all load ops.
  //  * `tileSizes`: An `ArrayRef<int64_t>` which gives the size of each
  //    index in the selected tiling. Uses the same order of indexes as in
  //    `perm` and `requirements`.
  // The `transform` function will also need to access the member variable
  // `op`, as this is the operation it is transforming.
  virtual void transform(const StencilOption &stencil,
                         mlir::ArrayRef<int64_t> tileSize) = 0;

  // Get the range of the `idx`th BlockArg
  int64_t getIdxRange(mlir::BlockArgument idx);

  // Print a log of the best stencil (reporting on cost, permutation, and
  // tiling) provided verbosity is at least `logLevel`
  void reportBestStencil(unsigned logLevel);

  // The number of indexes whose semantics must be considered in the tiling
  unsigned getTiledIdxCount() const { return requirements.size(); }

  // The BlockArguments of `op` (stored as a set for easy lookup)
  BlockArgumentSet getBlockArgsAsSet() const { return blockArgs; }

  // The ParallelOp that is being stenciled.
  mlir::AffineParallelOp op;

private:
  void bindIndexes(mlir::ArrayRef<ValueStrideInfo> values);
  void recursiveBindIndex(mlir::SetVector<mlir::BlockArgument> &b_idxs,
                          mlir::ArrayRef<ValueStrideInfo> values);
  void recursiveTileIndex(const StencilOption &stencil,
                          mlir::MutableArrayRef<int64_t> tileSize,
                          int64_t currIdx);
  StrideInfo getStrideInfo(mlir::Value value);

  // Cached call of the `idx`th tilingGenerator on parameter `range`
  std::vector<int64_t> generateTilings(int64_t idx, int64_t range);

  BlockArgumentSet blockArgs;

  mlir::SmallVector<StencilIndexRequirement> requirements;

  // Cache of results of tilingGenerators calls: First value of the key is which
  // generator was called, second value of the key is range it was called with
  mlir::DenseMap<std::pair<int64_t, int64_t>, std::vector<int64_t>>
      tilingsCache;

  // The range of each index (cached result of op.getConstantRanges())
  mlir::SmallVector<int64_t, 8> ranges;

  // The values captured by `capture.
  StencilCapture capturedValues;

  // Note: The bestCost, bestStencil, and bestTiling all must refer to the
  // same permutation & tiling choices and should only be modified together
  double bestCost;
  StencilOption bestStencil;
  mlir::SmallVector<int64_t, 8> bestTiling;

  util::ScheduleAttr schedule;
};

struct StencilCost {
  double throughput;
  unsigned startupCost;
};

using StencilCostFunction = std::function<StencilCost(
    mlir::ArrayRef<int64_t>, mlir::ArrayRef<mlir::Type>)>;

mlir::LogicalResult applyStencilGEMM(mlir::AffineParallelOp op,
                                     unsigned numThreads, bool isBatched,
                                     StencilCostFunction costFn);

mlir::AffineMap makeTileMap(mlir::MLIRContext *context, mlir::AffineMap map,
                            mlir::ValueRange operands,
                            mlir::ArrayRef<mlir::BlockArgument> idxs);

} // namespace pmlc::dialect::pxa
