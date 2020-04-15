// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/stencil_generic.h"

#include "pmlc/dialect/pxa/transforms/autotile.h" // TODO: for PowerOfTwoGenerator

#include "mlir/Support/DebugStringHelper.h" // TODO: sort

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
      return llvm::None;
    }

    // Find the Reduce Op
    auto it = std::prev(body->end(), 2);
    auto reduceOp = llvm::dyn_cast<AffineReduceOp>(*it);
    if (!reduceOp) {
      IVLOG(5, "The AffineParallelOp region didn't have a reduce as its last "
               "non-terminator");
      return llvm::None;
    }
    ret.stores.push_back(reduceOp);
    IVLOG(5, "Found ReduceOp");

    // Now check the reduceOp aggregation.
    if (reduceOp.agg() != AggregationKind::add) {
      IVLOG(5, "the reduce operation is not addition");
      return llvm::None;
    }

    // Get the operand for the reduce op and make sure it is the result of a
    // multiplication.
    auto defOp = reduceOp.val().getDefiningOp();
    if (!defOp) {
      IVLOG(5,
            "the source of the reduce operation is not defined in this block");
      return llvm::None;
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
      return llvm::None;
    }

    // Now verify the types of the operands of the mulOp must be affine.load
    // operations.
    if (!lhs || !rhs) {
      IVLOG(3, "the lhs or rhs of the mul operation are not affine.load "
               "operations.");
      return llvm::None;
    }
    ret.loads.push_back(lhs);
    ret.loads.push_back(rhs);

    return llvm::Optional<LoadStoreOps>(std::move(ret));
  }

  double getCost(TensorAndIndexPermutation perm, ArrayRef<int64_t> tileSize) {
    // TODO This is a fake cost function.
    // First, cap total tile size:
    int64_t totalTileSize = 1;
    for (auto sz : tileSize) {
      totalTileSize *= sz;
    }
    if (totalTileSize > 1024) {
      return std::numeric_limits<double>::infinity();
    }
    // Next, fewest tiles:
    int64_t tiles = 1;
    for (unsigned i = 0; i < semanticIdxCount; i++) {
      tiles *= llvm::divideCeil(getIdxRange(perm.indexes[i]), tileSize[i]);
    }
    return tiles;
  }

  void transform(TensorAndIndexPermutation perm, ArrayRef<int64_t> tileSize) {
    // TODO: Clean up this logging
    IVLOG(2, "Best Perf: " << bestCost);
    IVLOG(2, "Best Tensor/Index Permutations: TODO: print");
    std::stringstream bestTilingStr;
    bestTilingStr << "[ ";
    for (const auto &tileSize : bestTiling) {
      bestTilingStr << tileSize << " ";
    }
    bestTilingStr << "]";
    IVLOG(2, "Best Tiling: " << bestTilingStr.str());

    op.setAttr("is_gemm", mlir::UnitAttr::get(op.getContext()));
  }

public:
  explicit StencilXSMM(mlir::AffineParallelOp op) : StencilGeneric{op} {
    // TODO ctor
    // TODO: Probably want to move these to be params on StencilGeneric ctor...
    semanticIdxCount = 3; // TODO [i.e., must match generators & requirements]
    requirements =        // TODO: Make nicer
        std::map<std::pair<int64_t, int64_t>,
                 std::function<bool(mlir::Operation *, mlir::BlockArgument)>>{
            {{0, 0},
             [this](mlir::Operation *rawOp, mlir::BlockArgument a) {
               return getStrideInfo(rawOp)->strides[a] != 0;
             }},
            {{0, 1},
             [this](mlir::Operation *rawOp, mlir::BlockArgument a) {
               return getStrideInfo(rawOp)->strides[a] == 0;
             }},
            {{0, 2},
             [this](mlir::Operation *rawOp, mlir::BlockArgument a) {
               return getStrideInfo(rawOp)->strides[a] == 1;
             }},
            {{1, 0},
             [this](mlir::Operation *rawOp, mlir::BlockArgument a) {
               return getStrideInfo(rawOp)->strides[a] == 0;
             }},
            {{1, 1},
             [this](mlir::Operation *rawOp, mlir::BlockArgument a) {
               return getStrideInfo(rawOp)->strides[a] == 1;
             }},
            {{1, 2},
             [this](mlir::Operation *rawOp, mlir::BlockArgument a) {
               return getStrideInfo(rawOp)->strides[a] != 0;
             }},
            {{2, 0},
             [this](mlir::Operation *rawOp, mlir::BlockArgument a) {
               return getStrideInfo(rawOp)->strides[a] != 0;
             }},
            {{2, 1},
             [this](mlir::Operation *rawOp, mlir::BlockArgument a) {
               return getStrideInfo(rawOp)->strides[a] == 1;
             }},
            {{2, 2},
             [this](mlir::Operation *rawOp, mlir::BlockArgument a) {
               return getStrideInfo(rawOp)->strides[a] == 0;
             }},
        };
    tilingGenerators.push_back(PowerOfTwoGenerator());
    tilingGenerators.push_back(PowerOfTwoGenerator());
    tilingGenerators.push_back(PowerOfTwoGenerator());
  }
};

struct XSMMStencilPass
    : public mlir::PassWrapper<XSMMStencilPass, mlir::FunctionPass> {
  // I probably actually need config for requirements & tilingGenerators

  XSMMStencilPass() {}

  void runOnFunction() final {
    auto func = getFunction();
    // TODO: Capture `this` once pass has parameters
    func.walk([/*this*/](mlir::AffineParallelOp op) {
      StencilXSMM stencil(op);
      stencil.DoStenciling();
    });
  }
};

std::unique_ptr<mlir::Pass> createNewXSMMStencilPass() {
  return std::make_unique<XSMMStencilPass>();
}

} // namespace pmlc::dialect::pxa
