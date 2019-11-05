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
      float best_benefit = -1.0;
      ParallelForOp best_pf;
      int best_index = 0;
      int64_t best_split = 0;
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
        IVLOG(3,
              "  scale: " << scale << ", min: " << rem_range.min << ", range: " << idx_range << ", split: " << split);
        if (split <= 0 || split >= idx_range) {
          continue;
        }
        float benefit = scale > 0 ? idx_range - split : split;
        benefit /= idx_range;
        IVLOG(3, "  benefit = " << benefit)
        if (benefit > best_benefit) {
          best_benefit = benefit;
          best_pf = pf;
          best_index = arg->getArgNumber();
          best_split = split;
        }
      }
      if (best_benefit > 0) {
        found = true;
        // Make some changes starting at the parallel for
        auto builder = OpBuilder(best_pf);
        // Specifically, clone the pf
        auto copy_pf = mlir::cast<ParallelForOp>(builder.clone(*best_pf.getOperation()));
        LimitUpper(copy_pf, best_index, best_split);
        LimitLower(best_pf, best_index, best_split);
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
