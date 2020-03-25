// Copyright 2020 Intel Corporation

// TODO: includes etc
#include <functional>
#include <list>
#include <vector>

namespace pmlc::dialect::pxa {

struct TensorAndIndexPermutation{
  // An order of the Tensors and Indexes used in an operation
  llvm::SmallVector<Value, 3> tensors;
  std::vector<mlir::BlockArgument> indexes;
}

using StencilCaptureFcn = std::function<bool(
    const mlir::AffineParallelOp&,
    llvm::SmallVector<mlir::Op, 2>*,
    llvm::SmallVector<mlir::Op, 1>*,
    llvm::SmallVector<mlir::Op, 1>*,
    llvm::SmallVector<mlir::Op, 1>*)>;

using StencilPreflightFcn = std::function<std::list<TensorAndIndexPermutation>(
    const llvm::SmallVector<mlir::Op, 2>&,
    const llvm::SmallVector<mlir::Op, 1>&,
    const llvm::SmallVector<mlir::Op, 1>&,
    const llvm::SmallVector<mlir::Op, 1>&)>;

using Tiling = std::map<Value, unsigned>;

// TODO: Might be able to do a prettier generator than "return a list"?
using StencilTilingGeneratorFcn = std::function<std::list<Tiling>(
    const TensorAndIndexPermutation&)>;

using StencilCostFcn = std::function<double(
    const Tiling&,
    const TensorAndIndexPermutation&,
    const llvm::SmallVector<mlir::Op, 2>&,
    const llvm::SmallVector<mlir::Op, 1>&,
    const llvm::SmallVector<mlir::Op, 1>&,
    const llvm::SmallVector<mlir::Op, 1>&)>;

class StencilGeneric {
  // TODO
private:
  // The ParallelOp that is being stencilled.
  mlir::AffineParallelOp op;

  // The load operations
  llvm::SmallVector<mlir::Op, 2> load_ops;

  // // The (non-reduce) compute operations
  llvm::SmallVector<mlir::Op, 1> compute_ops;  // TODO: Needed?

  // The reduce operations
  llvm::SmallVector<mlir::Op, 1> reduce_ops;  // TODO: Needed?

  // The store operations
  llvm::SmallVector<mlir::Op, 1> store_ops;

  // The possibly legal permutations
  std::list<TensorAndIndexPermutation> permutations;

  // TODO: Some way of indicating none has been found, some way of tracking which TI permutation this goes with
  Tiling best_tiling;

  // Verifies pattern-match of ops and initializes load/reduce/store ops
  StencilCaptureFcn capture_fcn;

  // Pattern matches Tensor and Index access patterns, returns list of matching permutations
  StencilPreflightFcn preflight_fcn;

  // Produces Tilings to check for validity/optimality
  StencilTilingGeneratorFcn tiling_generator;

  // Computes the estimated cost of a tiling (infinity == invalid)
  StencilCostFcn cost_fcn;

  // After an optimal tiling has been found, apply it
  void Transform();
}


} // namespace pmlc::dialect::pxa
