// Copyright 2020 Intel Corporation

#pragma once

// TODO: includes etc
#include <algorithm>
#include <functional>
#include <limits>
#include <list>
#include <map>
#include <utility>
#include <vector>

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

namespace pmlc::dialect::pxa {

using BlockArgumentSet = llvm::SmallPtrSet<mlir::BlockArgument, 8>;

struct TensorAndIndexPermutation {
  // An order of the Tensors and Indexes used in an operation
  llvm::SmallVector<mlir::Operation *, 3> tensors;
  llvm::SmallVector<mlir::BlockArgument, 8> indexes;

  TensorAndIndexPermutation() = default;

  TensorAndIndexPermutation(llvm::SmallVector<mlir::Operation *, 3> tensors,
                            llvm::SmallVector<mlir::BlockArgument, 8> indexes)
      : tensors(tensors), indexes(indexes) {}
};

struct LoadStoreOps {
  // The load and store ops of an AffineParallel
  llvm::SmallVector<mlir::AffineLoadOp, 2> loads;
  llvm::SmallVector<AffineReduceOp, 1>
      stores; // TODO: Might be either store or reduce
};

// TODO: size_t or int64_t?
using TileSizeGenerator = std::function<std::vector<int64_t>(int64_t)>;

class StencilGeneric {
  // TODO
private:
  void BindIndexes(const llvm::SmallVector<mlir::Operation *, 3> &tensors);
  void RecursiveBindIndex(
      llvm::SmallVector<mlir::BlockArgument, 8> *bound_idxs,
      const llvm::SmallVector<mlir::Operation *, 3> &tensors);
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

  // The number of indexes whose semantics must be considered in the tiling
  unsigned semanticIdxCount; // TODO: how/where to initialize?

  // The ParallelOp that is being stenciled.
  mlir::AffineParallelOp op;

  // The BlockArguments of `op` (stored as a set for easy lookup)
  BlockArgumentSet blockArgs;

  // The load and store ops
  LoadStoreOps loadsAndStores;

  // The range of each index (cached result of op.getConstantRanges())
  llvm::SmallVector<int64_t, 8> ranges;

  // For each tensor/index semantic pair (given as a pair of size_ts), a
  // function to determine if a Value & BlockArg meet the requirements of that
  // pair
  std::map<std::pair<int64_t, int64_t>,
           std::function<bool(mlir::Operation *, mlir::BlockArgument)>>
      requirements;

  // For each semantically relevant index, a generator for tile sizes. Ordered
  // to match the index permutation.
  llvm::SmallVector<TileSizeGenerator, 5> tilingGenerators;

  double bestCost;
  TensorAndIndexPermutation bestPermutation;
  llvm::SmallVector<int64_t, 8>
      bestTiling; // only makes sense paired with `bestPermutation`

public:
  explicit StencilGeneric(mlir::AffineParallelOp op)
      : op(op), bestCost(std::numeric_limits<double>::infinity()) {
    for (auto blockArg : op.getBody()->getArguments()) {
      blockArgs.insert(blockArg);
    }
  }

  // Main function
  void DoStenciling();
};

} // namespace pmlc::dialect::pxa
