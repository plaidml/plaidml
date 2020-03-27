// Copyright 2020 Intel Corporation

// TODO: includes etc
#include <functional>
#include <list>
#include <vector>

namespace pmlc::dialect::pxa {

struct TensorAndIndexPermutation{
  // An order of the Tensors and Indexes used in an operation
  ArrayRef<Value> tensors;
  ArrayRef<mlir::BlockArgument> indexes;
}

struct LoadStoreOps{
  // The load and store ops of an AffineParallel
  llvm::SmallVector<mlir::Op, 2> loads;
  llvm::SmallVector<mlir::Op, 1> stores;
}

// TODO: Probably make some of these smart pointers

// TODO: May want something more like an autotile.h UnionGenerator?
using TileSizeGenerator = std::function<std::vector<int64_t>(int64_t)>;

using StencilTiling = std::map<mlir::BlockArgument, unsigned>;

// TODO: Here and everywhere else, I'm unclear whether I should be using
// mlir::Op or mlir::Op*
using StencilCaptureFn = std::function<Optional<LoadStoreOps>(
    mlir::AffineParallelOp)>;

using StencilPreflightFn = std::function<Optional<llvm::SmallVector<TileSizeGenerator, 15>>(
    mlir::AffineParallelOp,
    LoadStoreOps,
    TensorAndIndexPermutation)>;

using StencilCostFn = std::function<double(
    mlir::AffineParallelOp,
    LoadStoreOps,
    TensorAndIndexPermutation,
    StencilTiling)>;

class StencilGeneric {
  // TODO
private:
  // The ParallelOp that is being stencilled.
  mlir::AffineParallelOp op;

  // The load operations
  llvm::SmallVector<mlir::Op, 2> load_ops;

  // // The (non-reduce) compute operations
  llvm::SmallVector<mlir::Op, 1> compute_ops;  // TODO: Needed?

  // The reduce or store operations
  llvm::SmallVector<mlir::Op, 1> store_ops;

  // The possibly legal permutations
  std::list<TensorAndIndexPermutation> permutations;

  // TODO: Some way of indicating none has been found, some way of tracking which TI permutation this goes with
  StencilTiling best_tiling;

  // Verifies pattern-match of ops and initializes load/reduce/store ops
  StencilCaptureFn capture_fn;

  // Pattern matches Tensor and Index access patterns, returns list of matching permutations
  StencilPreflightFn preflight_fn;

  // Produces StencilTilings to check for validity/optimality
  StencilTilingGeneratorFn tiling_generator;

  // Computes the estimated cost of a tiling (infinity == invalid)
  StencilCostFn cost_fn;

  // After an optimal tiling has been found, apply it
  void Transform();

public:
  StencilGeneric(mlir::AffineParallelOp op,
                 StencilCaptureFn capture_fn,
                 StencilPreflightFn preflight_fn,
                 StencilTilingGeneratorFn tiling_generator,
                 StencilCostFn cost_fn)
      : op(op),
        capture_fn(capture_fn),
        preflight_fn(preflight_fn),
        tiling_generator(tiling_generator),
        cost_fn(cost_fn)
  {
    // TODO: Do nothing (?)
  }

  // Main function
  void DoStenciling();
}


} // namespace pmlc::dialect::pxa
