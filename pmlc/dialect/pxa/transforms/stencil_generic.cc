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

// TODO: Better to maintain bound_idxs as both set & vector, or just vector?
//   Or could maintain as map from BlockArg to numeric index (i.e. where it
//   would have been were it a vector)
// TODO: Also, should probably at least be a small vector (how small?)
void StencilGeneric::RecursiveBindIndex(
    const llvm::SmallVector<mlir::BlockArgument, 8> &bound_idxs,
    const llvm::SmallVector<mlir::Value, 3> &tensors) {
  auto curr_idx = bound_idxs.size();
  if (curr_idx == semantic_idx_count) {
    // This is a legal binding, append it and we're done with this branch
    legal_permutations.push_back(
        TensorAndIndexPermutation(tensors, bound_idxs));
  } else {
    for (const auto &block_arg : block_args) {
      // Don't bind same index twice
      if (std::find(bound_idxs.begin(), bound_idxs.end(), block_arg) !=
          bound_idxs.end()) {
        continue;
      }

      // Verify the requirements for this index with each tensor are all met
      bool reqs_met = true;
      for (size_t i = 0; i < tensors.size(); i++) {
        try { // TODO: probably don't keep long term
          if (!requirements.at(std::make_pair(i, curr_idx))(tensors[i],
                                                            block_arg)) {
            reqs_met = false;
            break;
          }
        } catch (const std::out_of_range &e) {
          IVLOG(1, "Error message: " << e.what());
          IVLOG(1, "Requested key was: " << std::make_pair(i, curr_idx));
          throw;
        }
      }
      if (!reqs_met) {
        continue;
      }

      // If we made it to here, this index has appropriate semantics; bind it
      // and recurse
      llvm::SmallVector<mlir::BlockArgument, 8> new_bound_idxs(bound_idxs);
      new_bound_idxs.push_back(block_arg);
      RecursiveBindIndex(new_bound_idxs, tensors);
    }
  }
}

void StencilGeneric::DoStenciling() {
  // Initialization
  auto maybe_ranges = op.getConstantRanges();
  if (maybe_ranges) {
    ranges = maybe_ranges.getValue(); // TODO: Is this how to use Optional?
  } else {
    IVLOG(4, "Cannot Stencil: Requires constant ranges");
    return;
  }

  auto maybe_loads_and_stores = capture();
  if (maybe_loads_and_stores) {
    loads_and_stores = maybe_loads_and_stores.getValue();
  } else {
    IVLOG(4, "Cannot Stencil: Operations fail to pattern-match.");
    return;
  }

  llvm::SmallVector<mlir::Value, 3> tensors;
  for (auto &load_op : loads_and_stores.loads) {
    tensors.push_back(load_op.getMemRef());
  }
  size_t first_store_idx = tensors.size();
  for (auto &store_op : loads_and_stores.stores) {
    tensors.push_back(store_op.out());
    // TODO: Probably should handle reduces vs. true stores in a different way
    // if (auto reduce_op = llvm::dyn_cast_or_null<AffineReduceOp>(store_op)) {
    //   tensors.push_back(reduce_op.out());
    // } else if (auto true_store_op =
    // llvm::dyn_cast_or_null<mlir::AffineStoreOp>(store_op)) {
    //   tensors.push_back(true_store_op.getMemRef());
    // } else {
    //   // TODO: throw?
    //   IVLOG(1, "Unexpected failure to load tensors from ops in stenciling");
    //   return;
    // }
  }
  auto last_load_first_store_it = tensors.begin() + first_store_idx;
  std::sort(tensors.begin(), last_load_first_store_it);
  do { // Each load tensor permutation
    std::sort(last_load_first_store_it, tensors.end());
    do { // Each store tensor permutation
      // Add all legal permutations to legal_permutations
      RecursiveBindIndex(llvm::SmallVector<mlir::BlockArgument, 8>(), tensors);
    } while (std::next_permutation(last_load_first_store_it, tensors.end()));
  } while (std::next_permutation(tensors.begin(), last_load_first_store_it));

  // TODO: If we desire more complex requirements than pairwise tensor-to-index
  // stride requirements, that function could go here

  for (const auto &perm : legal_permutations) {
    // TODO: Clean, try to get everything on the same unsigned type/width
    for (size_t i = 0; i < perm.indexes.size(); i++) {
      llvm::SmallVector<size_t, 8> tile_size;
      size_t idx_in_block = perm.indexes[i].getArgNumber();
      try { // TODO: probably don't keep long term
        for (const auto &size : tiling_generators[i](ranges[idx_in_block])) {
          tile_size.push_back(size);
        }
      } catch (const std::bad_function_call &e) {
        IVLOG(1, e.what());
        IVLOG(1, "Failed to find function for tiling_generators[" << i << "]");
        throw;
      }

      auto cost = getCost(perm, tile_size);
      if (cost < best_cost) {
        best_cost = cost;
        best_permutation = perm;
        best_tiling = std::move(tile_size);
      }
    }
  }

  if (best_cost < std::numeric_limits<double>::infinity()) {
    transform(best_permutation, best_tiling);
  } else {
    IVLOG(3, "No legal tiling found to stencil");
  }
}

} // namespace pmlc::dialect::pxa
