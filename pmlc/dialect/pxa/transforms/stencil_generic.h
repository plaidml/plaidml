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
  llvm::SmallVector<Value, 3> tensors;
  llvm::SmallVector<mlir::BlockArgument, 8> indexes;

  TensorAndIndexPermutation() = default;

  TensorAndIndexPermutation(llvm::SmallVector<Value, 3> tensors,
                            llvm::SmallVector<mlir::BlockArgument, 8> indexes)
      : tensors(tensors), indexes(indexes) {}
};

struct LoadStoreOps {
  // The load and store ops of an AffineParallel
  llvm::SmallVector<mlir::AffineLoadOp, 2> loads;
  llvm::SmallVector<AffineReduceOp, 1>
      stores; // TODO: Might be either store or reduce
};

// TODO: May want something more like an autotile.h UnionGenerator?
using TileSizeGenerator = std::function<std::vector<size_t>(size_t)>;

class StencilGeneric {
  // TODO
private:
  void RecursiveBindIndex(
      const llvm::SmallVector<mlir::BlockArgument, 8> &bound_idxs,
      const llvm::SmallVector<mlir::Value, 3> &tensors);

protected: // TODO: private backend for some of this?
  virtual llvm::Optional<LoadStoreOps> capture() = 0;
  // std::map<std::pair<size_t, size_t>, std::function<bool(mlir::Value,
  // mlir::BlockArgument)>> getRequirements();
  // llvm::SmallVector<TileSizeGenerator, 5> getGenerators();
  virtual double getCost(TensorAndIndexPermutation perm,
                         ArrayRef<size_t> tileSize) = 0;
  virtual void transform(TensorAndIndexPermutation perm,
                         ArrayRef<size_t> tileSize) = 0;

  // The number of indexes whose semantics must be considered in the tiling
  size_t semantic_idx_count; // TODO: how/where to initialize?

  // The ParallelOp that is being stenciled.
  mlir::AffineParallelOp op;

  // The BlockArguments of `op` (stored as a set for easy lookup)
  BlockArgumentSet block_args;

  // The load and store ops
  LoadStoreOps loads_and_stores;

  // For each tensor/index semantic pair (given as a pair of size_ts), a
  // function to determine if a Value & BlockArg meet the requirements of that
  // pair
  std::map<std::pair<size_t, size_t>,
           std::function<bool(mlir::Value, mlir::BlockArgument)>>
      requirements;

  // For each semantically relevant index, a generator for tile sizes. Ordered
  // to match the index permutation.
  llvm::SmallVector<TileSizeGenerator, 5> tiling_generators;

  double best_cost;
  TensorAndIndexPermutation best_permutation;
  llvm::SmallVector<size_t, 8>
      best_tiling; // only makes sense paired with `best_permutation`
  std::list<TensorAndIndexPermutation> legal_permutations;

public:
  explicit StencilGeneric(mlir::AffineParallelOp op)
      : op(op), best_cost(std::numeric_limits<double>::infinity()) {
    for (auto block_arg : op.getBody()->getArguments()) {
      block_args.insert(block_arg);
    }
  }

  // Main function
  void DoStenciling();
};

} // namespace pmlc::dialect::pxa
