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

class StencilGEMM : public StencilBase {
private:
  unsigned numThreads;
  bool doBatch;
  StencilCostFunction stencilCostFn;

  Optional<StencilCapture> capture() {
    using matchers::m_Any;
    // Looking for load..load..mul..reduce..terminator
    IVLOG(3, "StencilGEMMPass - in capture() function ");
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

    IVLOG(3, "StencilGEMMPass - returning NO match in capture() function ");
    return llvm::None;
  }

  // Requirements for additional reduction indices
  // that do not appear in output tensor
  IndexStridePredicates additionalReductionIdxReqs = IndexStridePredicates{
      [](int64_t stride) { return stride == 0; }, // output
      [](int64_t stride) { return stride != 0; }, // input0
      [](int64_t stride) { return stride != 0; }, // input1
  };

  bool isAdditionalReductionIndex(const BlockArgument &index,
                                  ArrayRef<Value> operands) {
    for (auto it : llvm::zip(operands, additionalReductionIdxReqs)) {
      Optional<StrideInfo> strideInfo = getStrideInfo(std::get<0>(it));
      int64_t stride = strideInfo->strides[index];
      if (!std::get<1>(it)(stride))
        return false;
    }
    return true;
  }

  double getCost(const StencilOption &stencil, ArrayRef<int64_t> tileSize) {
    unsigned tot_inner_loop = tileSize[0] * tileSize[1] * tileSize[2];

    SmallVector<Type, 3> types;
    for (Value value : stencil.values) {
      types.push_back(value.getType());
    }
    auto cost = stencilCostFn(tileSize, types);
    if (cost.throughput == 0) {
      return std::numeric_limits<double>::infinity();
    }
    double inner_time = tot_inner_loop / cost.throughput;
    IVLOG(6, "Inner loop (product of tile size) = " << tot_inner_loop);
    IVLOG(6, "Inner time (inner loop / throughput) = " << inner_time);
    for (unsigned i = 0; i < getTiledIdxCount(); ++i) {
      BlockArgument arg = stencil.indexes[i];
      IVLOG(6, debugString(arg) << ": " << tileSize[i]);
    }

    // The middle idxs are the accumulation indexes, i.e. those used on loads
    // but not stores
    DenseMap<BlockArgument, unsigned> middle_idxs;
    auto in0StrideInfo = getStrideInfo(stencil.values[1]);
    for (const auto &kvp : in0StrideInfo->strides) {
      if (getBlockArgsAsSet().count(kvp.first)) {
        IVLOG(6, "Based on first tensor, inserting middle index "
                     << kvp.first.getArgNumber());
        middle_idxs.insert(std::make_pair(kvp.first, getIdxRange(kvp.first)));
      } else {
        IVLOG(5, "Index found from outside current loop on left input: "
                     << kvp.first.getArgNumber());
      }
    }
    IVLOG(5, "Current size of middle_idxs = " << middle_idxs.size());

    auto in1StrideInfo = getStrideInfo(stencil.values[2]);
    for (const auto &kvp : in1StrideInfo->strides) {
      if (getBlockArgsAsSet().count(kvp.first)) {
        IVLOG(6, "Based on second tensor, inserting middle index "
                     << kvp.first.getArgNumber());
        middle_idxs.insert(std::make_pair(kvp.first, getIdxRange(kvp.first)));
      } else {
        IVLOG(5, "Index found from outside current loop on right input: "
                     << kvp.first.getArgNumber());
      }
    }
    IVLOG(5, "Current size of middle_idxs = " << middle_idxs.size());
    auto outStrideInfo = getStrideInfo(stencil.values[0]);
    for (const auto &kvp : outStrideInfo->strides) {
      if (getBlockArgsAsSet().count(kvp.first)) {
        auto it = middle_idxs.find(kvp.first);
        if (it != middle_idxs.end()) {
          IVLOG(6, "Based on output tensor, erasing middle index "
                       << it->first.getArgNumber());
          middle_idxs.erase(it);
        }
      } else {
        IVLOG(5, "Index found from outside current loop on output: "
                     << kvp.first.getArgNumber());
      }
    }

    for (unsigned i = 0; i < getTiledIdxCount(); ++i) {
      assert(getBlockArgsAsSet().count(stencil.indexes[i]) &&
             "All tiled indexes must be introduced in current loop");
      auto it = middle_idxs.find(stencil.indexes[i]);
      if (it != middle_idxs.end()) {
        it->second = llvm::divideCeil(it->second, tileSize[i]);
      }
    }
    unsigned tot_middle_loop = 1;
    for (auto &kvp : middle_idxs) {
      tot_middle_loop *= kvp.second;
    }

    IVLOG(4, "Middle: loop = " << tot_middle_loop);

    if (VLOG_IS_ON(4)) {
      for (auto &kvp : middle_idxs) {
        if (kvp.second > 1) {
          IVLOG(4, kvp.first.getArgNumber() << ": " << kvp.second);
        }
      }
    }

    DenseMap<BlockArgument, unsigned> outer_idxs;
    for (const auto &kvp : outStrideInfo->strides) {
      if (getBlockArgsAsSet().count(kvp.first)) {
        IVLOG(4, "First: " << kvp.first.getArgNumber());
        IVLOG(5, "Second: " << kvp.second);
        IVLOG(5, "IdxRange: " << getIdxRange(kvp.first));
        outer_idxs.try_emplace(kvp.first, getIdxRange(kvp.first));
        IVLOG(4, "And now emplaced");
      }
      // If the index is from outside `op` this has already been logged, no need
      // for `else` branch here
    }
    for (unsigned i = 0; i < getTiledIdxCount(); i++) {
      assert(getBlockArgsAsSet().count(stencil.indexes[i]) &&
             "All tiled indexes must be introduced in current loop");
      auto it = outer_idxs.find(stencil.indexes[i]);
      if (it != outer_idxs.end()) {
        it->second = llvm::divideCeil(it->second, tileSize[i]);
      }
    }
    unsigned tot_outer_loop = 1;
    for (auto &kvp : outer_idxs) {
      tot_outer_loop *= kvp.second;
    }

    IVLOG(4, "Outer: loop = " << tot_outer_loop);

    if (VLOG_IS_ON(4)) {
      for (auto &kvp : outer_idxs) {
        if (kvp.second > 1) {
          IVLOG(4, kvp.first.getArgNumber() << ": " << kvp.second);
        }
      }
    }

    unsigned outer_batches = (tot_outer_loop - 1) / numThreads + 1;
    double perf =
        outer_batches * tot_middle_loop * (cost.startupCost + inner_time);

    IVLOG(3, "Tile Information:");
    IVLOG(3, " tile:         "
                 << "[" << tileSize[0] << " " << tileSize[1] << " "
                 << tileSize[2] << "]");
    IVLOG(3, " outer count:  " << outer_batches);
    IVLOG(3, " middle count: " << tot_middle_loop);
    IVLOG(3, " startup cost: " << cost.startupCost);
    IVLOG(3, " inner time:   " << inner_time);
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
    auto opC = cast<PxaReduceOp>(stencil.values[0].getDefiningOp());
    auto opA = cast<PxaLoadOp>(stencil.values[1].getDefiningOp());
    auto opB = cast<PxaLoadOp>(stencil.values[2].getDefiningOp());

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
          isAdditionalReductionIndex(idx, ArrayRef<Value>{opC, opA, opB})) {
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
                    /*tilingGenerator=*/EvenTilingGenerator(),
                    /*predicates=*/IndexStridePredicates{
                        [](int64_t stride) { return stride != 0; }, // output
                        [](int64_t stride) { return stride != 0; }, // input0
                        [](int64_t stride) { return stride == 0; }, // input1
                    }},
                StencilIndexRequirement{
                    /*tilingGenerator=*/EvenTilingGenerator(),
                    /*predicates=*/IndexStridePredicates{
                        [](int64_t stride) { return stride == 1; }, // output
                        [](int64_t stride) { return stride == 0; }, // input0
                        [](int64_t stride) { return stride == 1; }, // input1
                    }},
                StencilIndexRequirement{
                    /*tilingGenerator=*/EvenTilingGenerator(),
                    /*predicates=*/IndexStridePredicates{
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
