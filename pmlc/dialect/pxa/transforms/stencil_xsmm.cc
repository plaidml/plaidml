// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/stencil_generic.h"

#include "pmlc/dialect/pxa/transforms/autotile.h" // TODO: for PowerOfTwoGenerator

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

// TODO: includes etc

namespace pmlc::dialect::pxa {

class StencilXSMM : public StencilGeneric {
private:
  llvm::Optional<LoadStoreOps> capture() {
    // Looking for load..load..mul..reduce..terminator
    LoadStoreOps ret;
    const unsigned kNumValidInstrInGemmRegion = 5;
    auto *body = op.getBody();

    // Verify the number of ops
    if (body->getOperations().size() != kNumValidInstrInGemmRegion) {
      IVLOG(5, "The AffineParallelOp region didn't have the right number of "
               "instructions for a GEMM");
      return llvm::Optional<LoadStoreOps>(); // i.e. fail to pattern-match
    }

    // Find the Reduce Op
    auto it = std::prev(body->end(), 2);
    auto reduceOp = llvm::dyn_cast<AffineReduceOp>(*it);
    if (!reduceOp) {
      IVLOG(5, "The AffineParallelOp region didn't have a reduce as its last "
               "non-terminator");
      return llvm::Optional<LoadStoreOps>(); // i.e. fail to pattern-match
    }
    ret.stores.push_back(reduceOp);
    IVLOG(5, "Found ReduceOp");

    // Now check the reduceOp aggregation.
    if (reduceOp.agg() != AggregationKind::add) {
      IVLOG(5, "the reduce operation is not addition");
      return llvm::Optional<LoadStoreOps>(); // i.e. fail to pattern-match
    }

    // Get the operand for the reduce op and make sure it is the result of a
    // multiplication.
    auto defOp = reduceOp.val().getDefiningOp();
    if (!defOp) {
      IVLOG(5,
            "the source of the reduce operation is not defined in this block");
      return llvm::Optional<LoadStoreOps>(); // i.e. fail to pattern-match
    }

    mlir::AffineLoadOp lhs;
    mlir::AffineLoadOp rhs;
    if (auto mulfOp = llvm::dyn_cast_or_null<mlir::MulFOp>(defOp)) {
      lhs = llvm::dyn_cast_or_null<mlir::AffineLoadOp>(
          mulfOp.lhs().getDefiningOp());
      rhs = llvm::dyn_cast_or_null<mlir::AffineLoadOp>(
          mulfOp.rhs().getDefiningOp());
    } else if (auto muliOp = llvm::dyn_cast_or_null<mlir::MulIOp>(defOp)) {
      lhs = llvm::dyn_cast_or_null<mlir::AffineLoadOp>(
          muliOp.lhs().getDefiningOp());
      rhs = llvm::dyn_cast_or_null<mlir::AffineLoadOp>(
          muliOp.rhs().getDefiningOp());
    } else {
      IVLOG(5, "The source of the reduce is not a multiplication operation");
      return llvm::Optional<LoadStoreOps>(); // i.e. fail to pattern-match
    }

    // Now verify the types of the operands of the mulOp must be affine.load
    // operations.
    if (!lhs || !rhs) {
      IVLOG(3, "the lhs or rhs of the mul operation are not affine.load "
               "operations.");
      return llvm::Optional<LoadStoreOps>(); // i.e. fail to pattern-match
    }
    ret.loads.push_back(lhs);
    ret.loads.push_back(rhs);

    return llvm::Optional<LoadStoreOps>(std::move(ret));
  }

  double getCost(TensorAndIndexPermutation perm, ArrayRef<size_t> tileSize) {
    // TODO: This is random garbage just to make some sort of test run (and
    // presumably fail)
    return 3;
  }

  void transform(TensorAndIndexPermutation perm, ArrayRef<size_t> tileSize) {
    IVLOG(2, "Best Perf: " << best_cost);
    IVLOG(2, "Best Tensor/Index Permutations: TODO: print");
    IVLOG(2, "Best Tiling: " << best_tiling[0]);

    op.setAttr("is_gemm", mlir::UnitAttr::get(op.getContext()));
  }

public:
  explicit StencilXSMM(mlir::AffineParallelOp op) : StencilGeneric{op} {
    // TODO ctor
    // TODO: Probably want to move these to be params on StencilGeneric ctor...
    semantic_idx_count = 1; // TODO
    requirements =
        std::map<std::pair<size_t, size_t>,
                 std::function<bool(mlir::Value, mlir::BlockArgument)>>{
            {{0, 0}, [](mlir::Value v, mlir::BlockArgument a) { return true; }},
            {{1, 0}, [](mlir::Value v, mlir::BlockArgument a) { return true; }},
            {{2, 0}, [](mlir::Value v, mlir::BlockArgument a) { return true; }},
            // TODO: Define `stride_of`...
            // {{0, 0}, [](mlir::Value v, mlir::BlockArgument a){ return
            // stride_of(v, a) != 0 }},
            // {{0, 1}, [](mlir::Value v, mlir::BlockArgument a){ return
            // stride_of(v, a) == 0 }},
            // {{0, 2}, [](mlir::Value v, mlir::BlockArgument a){ return
            // stride_of(v, a) == 1 }},
            // {{1, 0}, [](mlir::Value v, mlir::BlockArgument a){ return
            // stride_of(v, a) == 0 }},
            // {{1, 1}, [](mlir::Value v, mlir::BlockArgument a){ return
            // stride_of(v, a) == 1 }},
            // {{1, 2}, [](mlir::Value v, mlir::BlockArgument a){ return
            // stride_of(v, a) != 0 }},
            // {{2, 0}, [](mlir::Value v, mlir::BlockArgument a){ return
            // stride_of(v, a) != 0 }},
            // {{2, 1}, [](mlir::Value v, mlir::BlockArgument a){ return
            // stride_of(v, a) == 1 }},
            // {{2, 2}, [](mlir::Value v, mlir::BlockArgument a){ return
            // stride_of(v, a) == 0 }},
        };
    tiling_generators.push_back(PowerOfTwoGenerator());
  }
};

struct XSMMStencilPass : public mlir::FunctionPass<XSMMStencilPass> {
  // I probably actually need config for requirements & tiling_generators

  XSMMStencilPass() {}

  void runOnFunction() final {
    auto func = getFunction();
    func.walk([/*this*/](mlir::AffineParallelOp
                             op) { // TODO: Use `this` once pass has parameters
      StencilXSMM stencil(op);
      stencil.DoStenciling();
    });
  }
};

std::unique_ptr<mlir::Pass> createXSMMStencilPass() {
  return std::make_unique<XSMMStencilPass>();
}

} // namespace pmlc::dialect::pxa
