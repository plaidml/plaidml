// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/DebugStringHelper.h"

// TODO: Including autotile.h for PowerOfTwoGenerator, but maybe instead both
// should include a third file with the tile size generators
#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/autotile.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/pxa/transforms/stencil.h"
#include "pmlc/dialect/xsmm/ir/ops.h"

#include "pmlc/util/logging.h"

namespace pmlc::dialect::pxa {

class StencilXSMM : public StencilBase {
private:
  unsigned numThreads;
  StencilCostFunction stencilCostFn;

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
    ret.stores.push_back(&*it);
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

    mlir::Operation *lhs;
    mlir::Operation *rhs;
    if (auto mulfOp = llvm::dyn_cast_or_null<mlir::MulFOp>(defOp)) {
      lhs = mulfOp.lhs().getDefiningOp();
      if (!llvm::dyn_cast_or_null<mlir::AffineLoadOp>(lhs)) {
        IVLOG(3, "The LHS of the mul op is not affine.load.");
        return llvm::None;
      }
      rhs = mulfOp.rhs().getDefiningOp();
      if (!llvm::dyn_cast_or_null<mlir::AffineLoadOp>(rhs)) {
        IVLOG(3, "The RHS of the mul op is not affine.load.");
        return llvm::None;
      }
    } else if (auto muliOp = llvm::dyn_cast_or_null<mlir::MulIOp>(defOp)) {
      lhs = muliOp.lhs().getDefiningOp();
      if (!llvm::dyn_cast_or_null<mlir::AffineLoadOp>(lhs)) {
        IVLOG(3, "The LHS of the mul op is not affine.load.");
        return llvm::None;
      }
      rhs = muliOp.rhs().getDefiningOp();
      if (!llvm::dyn_cast_or_null<mlir::AffineLoadOp>(rhs)) {
        IVLOG(3, "The RHS of the mul op is not affine.load.");
        return llvm::None;
      }
    } else {
      IVLOG(5, "The source of the reduce is not a multiplication operation");
      return llvm::None;
    }
    ret.loads.push_back(lhs);
    ret.loads.push_back(rhs);

    return llvm::Optional<LoadStoreOps>(ret);
  }

  double getCost(TensorAndIndexPermutation perm, ArrayRef<int64_t> tileSize) {
    unsigned tot_inner_loop = tileSize[0] * tileSize[1] * tileSize[2];

    // Note: XSMM and its cost function heatmap use the reverse index order from
    // the rest of this code, hence the flip below
    llvm::SmallVector<unsigned, 3> xsmmTileSize;
    xsmmTileSize.push_back(tileSize[1]);
    xsmmTileSize.push_back(tileSize[0]);
    xsmmTileSize.push_back(tileSize[2]);
    auto cost = stencilCostFn(xsmmTileSize);
    if (cost.throughput == 0) {
      return std::numeric_limits<double>::infinity();
    }
    double inner_time = tot_inner_loop / cost.throughput;
    IVLOG(6,
          "Inner: loop = " << tot_inner_loop << " inner_time = " << inner_time);
    for (unsigned i = 0; i < getTiledIdxCount(); ++i) {
      IVLOG(6, perm.indexes[i] << ": " << tileSize[i]);
    }

    // The middle idxs are the accumulation indexes, i.e. those used on loads
    // but not stores
    llvm::DenseMap<mlir::BlockArgument, unsigned> middle_idxs;
    auto in0StrideInfo = getStrideInfo(perm.ioOps[1]);
    for (const auto &kvp : in0StrideInfo->strides) {
      if (getBlockArgsAsSet().count(kvp.first)) {
        IVLOG(6, "Based on first tensor, inserting middle index "
                     << kvp.first.getArgNumber());
        middle_idxs.insert(std::make_pair(kvp.first, getIdxRange(kvp.first)));
      } else {
        IVLOG(5, "Index found from outside current loop on left input: "
                     << kvp.first);
      }
    }
    IVLOG(5, "Current size of middle_idxs = " << middle_idxs.size());

    auto in1StrideInfo = getStrideInfo(perm.ioOps[2]);
    for (const auto &kvp : in1StrideInfo->strides) {
      if (getBlockArgsAsSet().count(kvp.first)) {
        IVLOG(6, "Based on second tensor, inserting middle index "
                     << kvp.first.getArgNumber());
        middle_idxs.insert(std::make_pair(kvp.first, getIdxRange(kvp.first)));
      } else {
        IVLOG(5, "Index found from outside current loop on right input: "
                     << kvp.first);
      }
    }
    IVLOG(5, "Current size of middle_idxs = " << middle_idxs.size());
    auto outStrideInfo = getStrideInfo(perm.ioOps[0]);
    for (const auto &kvp : outStrideInfo->strides) {
      if (getBlockArgsAsSet().count(kvp.first)) {
        auto it = middle_idxs.find(kvp.first);
        if (it != middle_idxs.end()) {
          IVLOG(6, "Based on output tensor, erasing middle index "
                       << it->first.getArgNumber());
          middle_idxs.erase(it);
        }
      } else {
        IVLOG(5,
              "Index found from outside current loop on output: " << kvp.first);
      }
    }

    for (unsigned i = 0; i < getTiledIdxCount(); ++i) {
      assert(getBlockArgsAsSet().count(perm.indexes[i]) &&
             "All tiled indexes must be introduced in current loop");
      auto it = middle_idxs.find(perm.indexes[i]);
      if (it != middle_idxs.end()) {
        it->second = llvm::divideCeil(it->second, tileSize[i]);
      }
    }
    unsigned tot_middle_loop = 1;
    for (auto &kvp : middle_idxs) {
      tot_middle_loop *= kvp.second;
    }

    IVLOG(4, "Middle: loop = " << tot_middle_loop);

    for (auto &kvp : middle_idxs) {
      if (kvp.second > 1) {
        IVLOG(4, kvp.first << ": " << kvp.second);
      }
    }

    llvm::DenseMap<mlir::BlockArgument, unsigned> outer_idxs;
    for (const auto &kvp : outStrideInfo->strides) {
      if (getBlockArgsAsSet().count(kvp.first)) {
        IVLOG(4, "First: " << kvp.first);
        IVLOG(5, "Second: " << kvp.second);
        IVLOG(5, "IdxRange: " << getIdxRange(kvp.first));
        outer_idxs.try_emplace(kvp.first, getIdxRange(kvp.first));
        IVLOG(4, "And now emplaced");
      }
      // If the index is from outside `op` this has already been logged, no need
      // for `else` branch here
    }
    for (unsigned i = 0; i < getTiledIdxCount(); i++) {
      assert(getBlockArgsAsSet().count(perm.indexes[i]) &&
             "All tiled indexes must be introduced in current loop");
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

    for (auto &kvp : outer_idxs) {
      if (kvp.second > 1) {
        IVLOG(4, kvp.first << ": " << kvp.second);
      }
    }

    unsigned outer_batches = (tot_outer_loop - 1) / numThreads + 1;
    double perf =
        outer_batches * tot_middle_loop * (cost.startupCost + inner_time);

    IVLOG(3, "Performance = " << perf << "(outer count: " << outer_batches
                              << ", middle count: " << tot_middle_loop
                              << ", startup cost: " << cost.startupCost
                              << ", inner time: " << inner_time << ")");
    return perf;
  }

  void transform(TensorAndIndexPermutation perm, ArrayRef<int64_t> tileSize) {
    // First, modify step size of all tiled indexes
    llvm::SmallVector<int64_t, 8> steps;
    auto oldSteps = op.steps().cast<ArrayAttr>().getValue();
    for (auto step : oldSteps) {
      steps.push_back(step.cast<IntegerAttr>().getInt());
    }
    for (size_t i = 0; i < getBlockArgsAsSet().size(); i++) {
      for (size_t j = 0; j < getTiledIdxCount(); j++) {
        if (perm.indexes[j] == op.getBody()->getArgument(i)) {
          steps[i] *= tileSize[j];
        }
      }
    }
    op.setSteps(steps);

    // Generate the XSMM call; first select inputs based on permutation order
    auto opC = llvm::dyn_cast<AffineReduceOp>(*perm.ioOps[0]);
    auto opA = llvm::dyn_cast<mlir::AffineLoadOp>(*perm.ioOps[1]);
    auto opB = llvm::dyn_cast<mlir::AffineLoadOp>(*perm.ioOps[2]);
    assert(opA && opB && opC);

    // Get the current memrefs
    Value aVal = opA.getMemRef();
    Value bVal = opB.getMemRef();
    Value cVal = opC.out();

    // Initialize helpers
    llvm::SmallVector<Value, 8> mapOperands;
    auto bodyBuilder = op.getBodyBuilder();
    auto makeTileMap = [&](AffineMap map, ValueRange ops,
                           ArrayRef<mlir::BlockArgument> idxs) {
      llvm::SmallVector<AffineExpr, 8> perOp;
      for (auto op : ops) {
        bool found = false;
        for (size_t i = 0; i < idxs.size(); i++) {
          if (op == idxs[i]) {
            perOp.push_back(bodyBuilder.getAffineDimExpr(i));
            found = true;
          }
        }
        if (!found) {
          perOp.push_back(bodyBuilder.getAffineConstantExpr(0));
        }
      }
      auto toIdxs = AffineMap::get(idxs.size(), 0, perOp, op.getContext());
      return map.compose(toIdxs);
    };

    // Set the tile size. Note XSMM wants n, m, k order and we have m, n, k
    llvm::SmallVector<int64_t, 3> xsmmTileSize;
    xsmmTileSize.push_back(tileSize[1]);
    xsmmTileSize.push_back(tileSize[0]);
    xsmmTileSize.push_back(tileSize[2]);
    auto tiles = bodyBuilder.getI64ArrayAttr(xsmmTileSize);

    // Set up the maps
    AffineMap cMap = opC.getAffineMap();
    AffineMap cTile = makeTileMap(opC.getAffineMap(), opC.getMapOperands(),
                                  {perm.indexes[0], perm.indexes[1]});
    mapOperands.append(opC.getMapOperands().begin(),
                       opC.getMapOperands().end());

    AffineMap aMap = opA.getAffineMap();
    AffineMap aTile = makeTileMap(opA.getAffineMap(), opA.getMapOperands(),
                                  {perm.indexes[0], perm.indexes[2]});
    mapOperands.append(opA.getMapOperands().begin(),
                       opA.getMapOperands().end());

    AffineMap bMap = opB.getAffineMap();
    AffineMap bTile = makeTileMap(opB.getAffineMap(), opB.getMapOperands(),
                                  {perm.indexes[2], perm.indexes[1]});
    mapOperands.append(opB.getMapOperands().begin(),
                       opB.getMapOperands().end());

    // Make the XSMM op
    bodyBuilder.create<xsmm::GemmOp>(op.getLoc(), cVal, cMap, cTile, aVal, aMap,
                                     aTile, bVal, bMap, bTile, tiles,
                                     mapOperands);

    // Remove all other ops from the op interior
    auto xsmm_it = std::prev(op.getBody()->end(), 2);
    while (op.getBody()->begin() != xsmm_it) {
      auto prev_it = std::prev(xsmm_it);
      op.getBody()->getOperations().erase(prev_it);
    }
  }

public:
  StencilXSMM(mlir::AffineParallelOp op, unsigned numThreads,
              StencilCostFunction costFn)
      : StencilBase{op,
                    3, // Three tileable indexes
                    {EvenTilingGenerator(), EvenTilingGenerator(),
                     EvenTilingGenerator()},
                    {IdxStrideReqs{
                         [](int64_t stride) { return stride != 0; }, // output
                         [](int64_t stride) { return stride != 0; }, // input0
                         [](int64_t stride) { return stride == 0; }, // input1
                     },
                     IdxStrideReqs{
                         [](int64_t stride) { return stride == 1; }, // output
                         [](int64_t stride) { return stride == 0; }, // input0
                         [](int64_t stride) { return stride == 1; }, // input1
                     },
                     IdxStrideReqs{
                         [](int64_t stride) { return stride == 0; }, // output
                         [](int64_t stride) { return stride == 1; }, // input0
                         [](int64_t stride) { return stride != 0; }, // input1
                     }}},
        numThreads{numThreads}, stencilCostFn(costFn) {}
};

struct XSMMStencilPass
    : public mlir::PassWrapper<XSMMStencilPass, mlir::FunctionPass> {
  // TODO: Do I want config for requirements & tilingGenerators?
  XSMMStencilPass() { assert(false && "XSMMStencilPass must be configured"); }

  XSMMStencilPass(const XSMMStencilPass &rhs) : costFn(rhs.costFn) {
    numThreads = rhs.numThreads.getValue();
  }

  XSMMStencilPass(unsigned numThreads_, StencilCostFunction costFn)
      : costFn(costFn) {
    numThreads = numThreads_;
  }

  void runOnFunction() final {
    auto func = getFunction();
    func.walk([this](mlir::AffineParallelOp op) {
      StencilXSMM stencil(op, numThreads.getValue(), costFn);
      stencil.DoStenciling();
    });
  }

  StencilCostFunction costFn;

  Option<unsigned> numThreads{
      *this, "threads",
      llvm::cl::desc("Specifies number of threads for the stencil pass")};
};

std::unique_ptr<mlir::Pass> createXSMMStencilPass(unsigned numThreads,
                                                  StencilCostFunction costFn) {
  return std::make_unique<XSMMStencilPass>(numThreads, costFn);
}

} // namespace pmlc::dialect::pxa
