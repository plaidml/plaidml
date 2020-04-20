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

#include "pmlc/target/x86/heatmap.h" // TODO: for heatmap

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

  // double getCost(TensorAndIndexPermutation perm, ArrayRef<int64_t> tileSize)
  // {
  //   // TODO This is a fake cost function.
  //   // First, cap total tile size:
  //   int64_t totalTileSize = 1;
  //   for (auto sz : tileSize) {
  //     totalTileSize *= sz;
  //   }
  //   if (totalTileSize > 1024) {
  //     return std::numeric_limits<double>::infinity();
  //   }
  //   // Next, fewest tiles:
  //   int64_t tiles = 1;
  //   for (unsigned i = 0; i < semanticIdxCount; i++) {
  //     tiles *= llvm::divideCeil(getIdxRange(perm.indexes[i]), tileSize[i]);
  //   }
  //   return tiles;
  // }

  double getCost(TensorAndIndexPermutation perm, ArrayRef<int64_t> tileSize) {
    unsigned numThreads = 4; // TODO

    unsigned tot_inner_loop = tileSize[0] * tileSize[1] * tileSize[2];

    llvm::SmallVector<unsigned, 3> tileSizeTODO;
    for (unsigned i = 0; i < 3; ++i) {
      tileSizeTODO.push_back(tileSize[i]);
    }
    auto cost = pmlc::target::x86::heatmapCost(tileSizeTODO);
    if (cost.throughput == 0) {
      return std::numeric_limits<double>::infinity();
    }
    double inner_time = tot_inner_loop / cost.throughput;
    IVLOG(6,
          "Inner: loop = " << tot_inner_loop << " inner_time = " << inner_time);
    for (unsigned i = 0; i < semanticIdxCount; ++i) {
      IVLOG(6, perm.indexes[i] << ": " << tileSize[i]);
    }
    // return 3; // TODO cheap hack  !!!!! Works on all verbosities if I return here

    // The middle idxs are the accumulation indexes, i.e. those used on loads but not stores
    // llvm::DenseMap<mlir::BlockArgument, unsigned> middle_idxs;
    std::map<mlir::BlockArgument, unsigned> middle_idxs;  // TODO: Why does this matter?
    for (const auto& kvp : getStrideInfo(perm.tensors[0])->strides) {
      // TODO: Old version verifies that this is in the parallel op's BlockArgs, but that seems excessive for something that I'd expect to be an assert...
      if (blockArgs.find(kvp.first) == blockArgs.end()) {
        IVLOG(1, "Hey this isn't a real block arg!: " << kvp.first);
        // throw std::runtime_error("TODO Guess we do need to check!");
      } else {
        middle_idxs.insert(std::make_pair(kvp.first, getIdxRange(kvp.first)));
      }
      // return 3; // TODO: cheap hack...  !!!!!!!! Returning here gives intermittent verbosity 3 errors
    }
    // return 3;  // TODO: cheap hack  !!!!!!!!!!!! Breaks on verbosity 3+ if I return here
    for (const auto& kvp : getStrideInfo(perm.tensors[1])->strides) {
      // TODO: Old version verifies that this is in the parallel op's BlockArgs, but that seems excessive for something that I'd expect to be an assert...
      middle_idxs.insert(std::make_pair(kvp.first, getIdxRange(kvp.first)));
    }
    for (const auto &kvp : getStrideInfo(perm.tensors[2])->strides) {
      auto it = middle_idxs.find(kvp.first);
      if (it != middle_idxs.end()) {
        middle_idxs.erase(it);
      }
    }
    // return 3;  // TODO: cheap hack

    for (unsigned i = 0; i < semanticIdxCount; ++i) {
      auto it = middle_idxs.find(perm.indexes[i]);
      if (it != middle_idxs.end()) {
        it->second = llvm::divideCeil(it->second, tileSize[i]);
      }
    }
    unsigned tot_middle_loop = 1;
    for (auto &kvp : middle_idxs) {
      tot_middle_loop *= kvp.second;
    }
    // return 3;  // TODO: cheap hack !!!! INTERMITTENTLY breaks on verbosity 2 if I return here

    IVLOG(4, "Middle: loop = " << tot_middle_loop);

    for (auto &kvp : middle_idxs) {
      if (kvp.second > 1) {
        IVLOG(4, kvp.first << ": " << kvp.second);
      }
    }
    // return 3; // TODO: less cheap hack

    // ... TODO unclear of port quality
    // llvm::DenseMap<mlir::BlockArgument, unsigned> outer_idxs;
    std::map<mlir::BlockArgument, unsigned> outer_idxs;  // TODO why does this matter...
    for (const auto& kvp : getStrideInfo(loadsAndStores.stores[0])->strides) {
      IVLOG(4, "First: " << kvp.first);
      IVLOG(5, "Second: " << kvp.second);
      IVLOG(5, "IdxRange: " << getIdxRange(kvp.first));
      outer_idxs.try_emplace(kvp.first, getIdxRange(kvp.first));
      IVLOG(4, "And now emplaced");
    }
    IVLOG(4, "Left loop...");
    // return 3; // TODO: less cheap hack
    for (unsigned i = 0; i < semanticIdxCount; i++) {
      auto it = outer_idxs.find(perm.indexes[i]);
      if (it != outer_idxs.end()) {
        it->second = llvm::divideCeil(it->second, tileSize[i]);
      }
    }
    unsigned tot_outer_loop = 1;
    for (auto &kvp : outer_idxs) {
      tot_outer_loop *= kvp.second;
    }

    IVLOG(4, "Outer: loop = " << tot_outer_loop);

    // llvm::DenseMap<mlir::BlockArgument, unsigned> outer_idxs;
    // for (auto idx : outIdxs) {
    //   outer_idxs.try_emplace(idx, idxRange(idx));
    // }
    // for (unsigned i = 0; i < semanticIdxCount; ++i) {
    //   auto it = outer_idxs.find(innerIdxs[i]);
    //   if (it != outer_idxs.end()) {
    //     it->second = (it->second - 1) / tileSize[i] + 1;
    //   }
    // }
    // unsigned tot_outer_loop = 1;
    // for (auto &kvp : outer_idxs) {
    //   tot_outer_loop *= kvp.second;
    // }

    // IVLOG(3, "Outer: loop = " << tot_outer_loop);

    for (auto &kvp : outer_idxs) {
      if (kvp.second > 1) {
        IVLOG(4, kvp.first << ": " << kvp.second);
      }
    }

    unsigned outer_batches = (tot_outer_loop - 1) / numThreads + 1;
    double perf =
        outer_batches * tot_middle_loop * (cost.startupCost + inner_time);

    IVLOG(4, "Performance = " << perf);
    return perf;
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
    tilingGenerators.push_back(EvenTilingGenerator());
    tilingGenerators.push_back(EvenTilingGenerator());
    tilingGenerators.push_back(EvenTilingGenerator());
  }
};

struct NewXSMMStencilPass
    : public mlir::PassWrapper<NewXSMMStencilPass, mlir::FunctionPass> {
  // I probably actually need config for requirements & tilingGenerators

  NewXSMMStencilPass() {}

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
  return std::make_unique<NewXSMMStencilPass>();
}

} // namespace pmlc::dialect::pxa
