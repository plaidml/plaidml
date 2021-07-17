// Copyright 2020 Intel Corporation

#include <string>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/ir/matchers.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/autotile.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/pxa/transforms/stencil.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/matchers.h"
#include "pmlc/util/tags.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

struct GemmOperand {
  Value memref;
  AffineMap accessMap;
  AffineMap tileMap;

  template <typename TOp>
  GemmOperand(TOp op, ArrayRef<BlockArgument> idxs,
              SmallVectorImpl<Value> &mapOperands)
      : memref(op.getMemRef()), accessMap(op.getAffineMap()),
        tileMap(makeTileMap(op.getContext(), op.getAffineMap(),
                            op.getMapOperands(), idxs)) {
    mapOperands.append(op.getMapOperands().begin(), op.getMapOperands().end());
  }
};

// Requirements for additional reduction indices that do not appear in output
// tensor
static StencilIndexRequirement etcIdxReqs{
    /*idxName=*/"etc",                         // this is unused
    /*tilingGenerator=*/EvenTilingGenerator(), // this is unused
    IndexStridePredicates{
        [](int64_t stride) { return stride == 0; }, // output
        [](int64_t stride) { return stride != 0; }, // input0
        [](int64_t stride) { return stride != 0; }, // input1
    }};

class StencilGEMM : public StencilBase {
private:
  unsigned numThreads;
  bool doBatch;
  StencilCostFunction stencilCostFn;

  Optional<StencilCapture> capture() {
    using matchers::m_Any;
    // Looking for load..load..mul..reduce..terminator
    Value load1, load2, reduce;
    Operation *yield = op.getBody()->getTerminator();
    if (matchPattern(
            yield,
            m_Op<AffineYieldOp>(m_Capture(
                &reduce, m_PxaReduceOp(
                             AtomicRMWKind::addf,
                             m_Op<MulFOp>(m_Capture(&load1, m_Op<PxaLoadOp>()),
                                          m_Capture(&load2, m_Op<PxaLoadOp>())),
                             m_Any())))) ||
        matchPattern(
            yield,
            m_Op<AffineYieldOp>(m_Capture(
                &reduce, m_PxaReduceOp(
                             AtomicRMWKind::addi,
                             m_Op<MulIOp>(m_Capture(&load1, m_Op<PxaLoadOp>()),
                                          m_Capture(&load2, m_Op<PxaLoadOp>())),
                             m_Any()))))) {
      return StencilCapture{{reduce}, {load1, load2}};
    }
    return llvm::None;
  }

  double getCost(const StencilOption &stencil, ArrayRef<int64_t> tileSize) {
    SmallVector<Type, 3> types;
    for (const ValueStrideInfo &vsi : stencil.values) {
      types.push_back(vsi.value.getType());
    }
    auto cost = stencilCostFn(tileSize, types);
    if (cost.throughput == 0) {
      return std::numeric_limits<double>::infinity();
    }

    // The middle idxs are the accumulation indexes, i.e. those used on loads
    // but not stores
    DenseMap<BlockArgument, unsigned> middleIdxs, outerIdxs;
    for (const auto &kvp : stencil.values[1].strideInfo.strides) {
      if (getBlockArgsAsSet().count(kvp.first)) {
        IVLOG(6, "Based on first tensor, inserting middle index "
                     << kvp.first.getArgNumber());
        middleIdxs.insert(std::make_pair(kvp.first, getIdxRange(kvp.first)));
      }
    }
    IVLOG(5, "Current size of middleIdxs = " << middleIdxs.size());

    for (const auto &kvp : stencil.values[2].strideInfo.strides) {
      if (getBlockArgsAsSet().count(kvp.first)) {
        IVLOG(6, "Based on second tensor, inserting middle index "
                     << kvp.first.getArgNumber());
        middleIdxs.insert(std::make_pair(kvp.first, getIdxRange(kvp.first)));
      }
    }
    IVLOG(5, "Current size of middleIdxs = " << middleIdxs.size());

    for (const auto &kvp : stencil.values[0].strideInfo.strides) {
      if (getBlockArgsAsSet().count(kvp.first)) {
        auto it = middleIdxs.find(kvp.first);
        if (it != middleIdxs.end()) {
          IVLOG(6, "Based on output tensor, erasing middle index "
                       << it->first.getArgNumber());
          middleIdxs.erase(it);
        }
        outerIdxs.try_emplace(kvp.first, getIdxRange(kvp.first));
      }
    }

    for (unsigned i = 0; i < getTiledIdxCount(); ++i) {
      assert(getBlockArgsAsSet().count(stencil.indexes[i]) &&
             "All tiled indexes must be introduced in current loop");

      auto itMiddle = middleIdxs.find(stencil.indexes[i]);
      if (itMiddle != middleIdxs.end()) {
        itMiddle->second = llvm::divideCeil(itMiddle->second, tileSize[i]);
      }

      auto itOuter = outerIdxs.find(stencil.indexes[i]);
      if (itOuter != outerIdxs.end()) {
        itOuter->second = llvm::divideCeil(itOuter->second, tileSize[i]);
      }
    }

    unsigned totOuterLoop = 1;
    for (const auto &kvp : outerIdxs) {
      totOuterLoop *= kvp.second;
    }

    unsigned totMiddleLoop = 1;
    for (const auto &kvp : middleIdxs) {
      totMiddleLoop *= kvp.second;
    }

    unsigned totInnerLoop = tileSize[0] * tileSize[1] * tileSize[2];
    double innerTime = totInnerLoop / cost.throughput;

    IVLOG(4, "Outer: loop = " << totOuterLoop);
    if (VLOG_IS_ON(4)) {
      for (auto &kvp : outerIdxs) {
        if (kvp.second > 1) {
          IVLOG(4, kvp.first.getArgNumber() << ": " << kvp.second);
        }
      }
    }

    unsigned outerBatches = (totOuterLoop - 1) / numThreads + 1;
    double perf = outerBatches * totMiddleLoop * (cost.startupCost + innerTime);

    IVLOG(3, "Tile Information:");
    IVLOG(3, " tile:         "
                 << "[" << tileSize[0] << " " << tileSize[1] << " "
                 << tileSize[2] << "]");
    IVLOG(3, " outer count:  " << outerBatches);
    IVLOG(3, " middle count: " << totMiddleLoop);
    IVLOG(3, " startup cost: " << cost.startupCost);
    IVLOG(3, " inner time:   " << innerTime);
    return perf;
  }

  void transform(const StencilOption &stencil, ArrayRef<int64_t> tileSize) {
    int64_t kBatches = 1;
    int64_t kRange = getIdxRange(stencil.indexes[2]);
    IVLOG(3, "kRange: " << kRange);
    SmallVector<BlockArgument> tileEtcIdxs(stencil.indexes.begin(),
                                           stencil.indexes.end());
    SmallVector<int64_t> batches;

    // Generate the GEMM op; select inputs based on permutation order
    auto opC = cast<PxaReduceOp>(stencil.values[0].value.getDefiningOp());
    auto opA = cast<PxaLoadOp>(stencil.values[1].value.getDefiningOp());
    auto opB = cast<PxaLoadOp>(stencil.values[2].value.getDefiningOp());

    // First, modify step size of all tiled indexes
    SmallVector<int64_t, 8> steps = op.getSteps();
    for (size_t i = 0; i < steps.size(); i++) {
      BlockArgument idx = op.getBody()->getArgument(i);
      int64_t idxRange = getIdxRange(idx);

      bool foundBlockArg = false;
      for (size_t j = 0; j < getTiledIdxCount(); j++) {
        if (stencil.indexes[j] == idx) {
          steps[i] *= tileSize[j];
          foundBlockArg = true;

          // K index (reduction dimension)
          if (doBatch && j == 2) {
            // We want to transform "regular" pxa.generic where kBatches is 1:
            // affine.parallel (i, j, k) = (..., 0) to (..., kRange)
            //                             step (kStep)
            //   pxa.generic C[i, j] = A[i, k], B[k, j]: [..., kStep], 1
            //
            // to
            //
            // affine.parallel (i, j, k) = (..., 0) to (..., kRange)
            //                             step (kRange)
            //   pxa.generic C[i, j] = A[i, k], B[k, j]: [..., kStep],
            //    (kRange/64)
            //
            // where the number of batches of A and B matrices to multiply
            // is the k loop's range divided by the original step size for
            // the k loop.
            //
            // Subsequently, kStep is set to kRange. That is, in one step, a
            // block of C is completely computed through reduction of
            // batches of A and B matrix multiplies.
            kBatches = kRange / steps[i];
            steps[i] = kRange;

            IVLOG(3, "steps[" << i << "] = " << steps[i]);
            IVLOG(3, "kBatches: " << kBatches);
          }
        }
      }

      // Check for additional reduction indices with a range greater than 1
      if (doBatch && !foundBlockArg && steps[i] == 1 && idxRange > 1 &&
          etcIdxReqs.check(stencil.values, idx)) {
        tileEtcIdxs.push_back(idx);
        batches.emplace_back(idxRange);
        foundBlockArg = true;
        steps[i] = idxRange;
      }
    }
    op.setSteps(steps);

    // fullTileSizes always has the form: [m, n, k, kBatches, others...]
    SmallVector<int64_t> fullTileSizes;
    fullTileSizes.append(tileSize.begin(), tileSize.end());
    if (!batches.empty()) {
      fullTileSizes.push_back(kBatches);
      fullTileSizes.append(batches.begin(), batches.end());
    } else if (kBatches != 1) {
      fullTileSizes.push_back(kBatches);
    }

    OpBuilder builder = op.getBodyBuilder();

    SmallVector<Value> inputIndices, outputIndices;
    GemmOperand c(opC, stencil.indexes, outputIndices);
    GemmOperand a(opA, tileEtcIdxs, inputIndices);
    GemmOperand b(opB, tileEtcIdxs, inputIndices);

    ArrayAttr outputAccessMaps = builder.getAffineMapArrayAttr({c.accessMap});
    ArrayAttr outputTileMaps = builder.getAffineMapArrayAttr({c.tileMap});

    ArrayAttr inputAccessMaps =
        builder.getAffineMapArrayAttr({a.accessMap, b.accessMap});
    ArrayAttr inputTileMaps =
        builder.getAffineMapArrayAttr({a.tileMap, b.tileMap});

    ArrayAttr reductions =
        builder.getI64ArrayAttr({static_cast<int64_t>(opC.agg())});

    auto genericOp = builder.create<PxaGenericOp>(
        op.getLoc(), c.memref.getType(),
        /*inputs=*/ArrayRef<Value>{a.memref, b.memref},
        /*outputs=*/ArrayRef<Value>{c.memref},
        /*inputIndices=*/inputIndices,
        /*outputIndices=*/outputIndices,
        /*inputAccessMaps=*/inputAccessMaps,
        /*inputTileMaps=*/inputTileMaps,
        /*outputAccessMaps=*/outputAccessMaps,
        /*outputTileMaps=*/outputTileMaps,
        /*kernel=*/builder.getStringAttr("tpp_gemm"),
        /*tile=*/builder.getI64ArrayAttr(fullTileSizes),
        /*reductions=*/reductions);

    opC.result().replaceAllUsesWith(genericOp.getResult(0));
    opC.erase();
  }

public:
  StencilGEMM(AffineParallelOp op, unsigned numThreads, bool doBatch,
              StencilCostFunction costFn)
      : StencilBase(
            op,
            {
                StencilIndexRequirement{
                    /*idxName=*/"gemm_m",
                    /*tilingGenerator=*/EvenTilingGenerator(),
                    IndexStridePredicates{
                        [](int64_t stride) { return stride != 0; }, // output
                        [](int64_t stride) { return stride != 0; }, // input0
                        [](int64_t stride) { return stride == 0; }, // input1
                    }},
                StencilIndexRequirement{
                    /*idxName=*/"gemm_n",
                    /*tilingGenerator=*/EvenTilingGenerator(),
                    IndexStridePredicates{
                        [](int64_t stride) { return stride == 1; }, // output
                        [](int64_t stride) { return stride == 0; }, // input0
                        [](int64_t stride) { return stride == 1; }, // input1
                    }},
                StencilIndexRequirement{
                    /*idxName=*/"gemm_k",
                    /*tilingGenerator=*/EvenTilingGenerator(),
                    IndexStridePredicates{
                        [](int64_t stride) { return stride == 0; }, // output
                        [](int64_t stride) { return stride == 1; }, // input0
                        [](int64_t stride) { return stride != 0; }, // input1
                    }},
            }),
        numThreads{numThreads}, doBatch{doBatch}, stencilCostFn(costFn) {}
};

LogicalResult applyStencilGEMM(AffineParallelOp op, unsigned numThreads,
                               bool doBatch, StencilCostFunction costFn) {
  StencilGEMM stencil(op, numThreads, doBatch, costFn);
  stencil.performStenciling();
  return success();
}

} // namespace pmlc::dialect::pxa
