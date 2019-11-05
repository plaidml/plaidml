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

struct JigsawPass : public mlir::FunctionPass<JigsawPass> {
  void runOnFunction() override;
};

void JigsawPass::runOnFunction() {
  mlir::FuncOp f = getFunction();
  std::cerr << "WOOT!\n";

  Block* b = &f.getBody().front();
  auto it_prev = b->end();
  auto it_end = std::prev(b->end(), 1);
  while (true) {
    auto it = std::next(it_prev, 1);
    if (it == it_end) {
      break;
    }
    bool found = false;
    it->walk([&](ConstraintOp op) {
      AffinePolynomial poly(op.input());
      int64_t best_benifit = -1;
      ParallelForOp best_pf;
      int best_index;
      int64_t best_split;
      bool best_pos;
      for (auto [arg, scale] : poly.terms) {
        // Get the range of the index at hand
        auto pf = mlir::cast<ParallelForOp>(arg->getOwner()->getParentOp());
        int64_t idx_range = pf.getRange(arg->getArgNumber());
        // Make a version of the poly remainder without the current argument
        auto rem_poly = poly;
        rem_poly.terms.erase(arg);
        // Get the range of the remainder
        AffineRange rem_range(rem_poly);
        int64_t split = scale > 0 ? (-rem_range.min + scale - 1) / scale : (-rem_range.min / scale) + 1;
        std::cerr << "  scale: " << scale << ", min: " << rem_range.min << ", max: " << rem_range.max
                  << ", range: " << idx_range << "\n";
        std::cerr << "  split: " << split << "\n";
        if (split <= 0 || split >= idx_range) {
          continue;
        }
        int64_t benifit = scale > 0 ? idx_range - split : split;
        if (benifit > best_benifit) {
          best_benifit = benifit;
          best_pf = pf;
          best_index = arg->getArgNumber();
          best_split = split;
          best_pos = (scale > 0);
        }
      }
      if (best_benifit > 0) {
        std::cerr << "best_benifit: " << best_benifit << ", best_split: " << best_split << ", best_pos: " << best_pos
                  << "\n";
        found = true;
        // Make some changes starting at the parallel for
        auto builder = OpBuilder(best_pf);
        // Specifically, clone the pf
        auto copy_pf = mlir::cast<ParallelForOp>(builder.clone(*best_pf.getOperation()));
        LimitLower(copy_pf, best_index, best_split);
        LimitUpper(best_pf, best_index, best_split);
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (found) {
      // Try again at the current place
    } else {
      it_prev++;
    }
  }
}

static mlir::PassRegistration<JigsawPass> jigsaw_pass("stripe-jigsaw",
                                                      "Split parallel-fors into bits to remove constraints");

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
