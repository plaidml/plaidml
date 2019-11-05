// Copyright 2019, Intel Corporation

// A very naive vectorization pass as a demo of Stripe dialect

#include "base/util/logging.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "pmlc/dialect/stripe/analysis.h"
#include "pmlc/dialect/stripe/ops.h"
#include "pmlc/dialect/stripe/rewrites.h"
#include "pmlc/dialect/stripe/transforms.h"

namespace pmlc {
namespace dialect {
namespace stripe {

struct VectorizePass : public mlir::FunctionPass<VectorizePass> {
  void runOnFunction() override;
};

// Get all of the parallel for ops, pick a dimension to vectorize on, and
// vectorize.  To handle uneven splits, add additional constraints in the
// inner loop (which would be later removed via contraint lifting)
void VectorizePass::runOnFunction() {
  mlir::FuncOp f = getFunction();
  f.walk([](ParallelForOp op) {
    // Compute tiling sizes (presume first element is the one to vectorize over)
    llvm::SmallVector<int64_t, 8> tile_sizes;
    for (size_t i = 0; i < op.ranges().size(); i++) {
      if (i == 0) {
        tile_sizes.push_back(32);
      } else {
        tile_sizes.push_back(1);
      }
    }
    // Do the tiling
    Tile(op, tile_sizes);
  });
}

static mlir::PassRegistration<VectorizePass> vectorize_pass("stripe-vectorize", "Vectorize a stripe program");

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
