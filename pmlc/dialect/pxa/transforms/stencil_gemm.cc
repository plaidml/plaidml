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

static AffineMap makeTileMap(MLIRContext *context, AffineMap map,
                             ValueRange operands,
                             ArrayRef<BlockArgument> idxs) {
  SmallVector<AffineExpr, 8> exprs;
  for (auto value : operands) {
    bool found = false;
    for (size_t i = 0; i < idxs.size(); i++) {
      if (value == idxs[i]) {
        exprs.push_back(getAffineDimExpr(i, context));
        found = true;
      }
    }
    if (!found) {
      exprs.push_back(getAffineConstantExpr(0, context));
    }
  }
  auto toIdxs = AffineMap::get(idxs.size(), 0, exprs, context);
  return map.compose(toIdxs);
}

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

  Optional<LoadStoreOps> capture() {
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
      return LoadStoreOps{{reduce}, {load1, load2}};
    }
    return llvm::None;
  }

  // Requirements for additional reduction indices
  // that do not appear in output tensor
  IdxStrideReqs additionalReductionIdxReqs = IdxStrideReqs{
      [](int64_t stride) { return stride == 0; }, // output
      [](int64_t stride) { return stride != 0; }, // input0
      [](int64_t stride) { return stride != 0; }, // input1
  };

  bool isAdditionalReductionIndex(const BlockArgument &index,
                                  ArrayRef<Value> tensor) {
    for (size_t i = 0; i < tensor.size(); i++) {
      auto strideInfo = getStrideInfo(tensor[i]);
      auto stride = strideInfo->strides[index];
      if (!additionalReductionIdxReqs[i](stride)) {
        return false;
      }
    }
    return true;
  }

  double getCost(TensorAndIndexPermutation perm, ArrayRef<int64_t> tileSize) {
    unsigned tot_inner_loop = tileSize[0] * tileSize[1] * tileSize[2];

    SmallVector<Type, 3> types;
    for (Value value : perm.values) {
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
      IVLOG(6, debugString(perm.indexes[i]) << ": " << tileSize[i]);
    }

    // The middle idxs are the accumulation indexes, i.e. those used on loads
    // but not stores
    DenseMap<BlockArgument, unsigned> middle_idxs;
    auto in0StrideInfo = getStrideInfo(perm.values[1]);
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

    auto in1StrideInfo = getStrideInfo(perm.values[2]);
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
    auto outStrideInfo = getStrideInfo(perm.values[0]);
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

  void transform(TensorAndIndexPermutation perm, ArrayRef<int64_t> tileSize) {
    int64_t kBatches = 1;
    int64_t kRange = getIdxRange(perm.indexes[2]);
    IVLOG(3, "kRange: " << kRange);
    SmallVector<BlockArgument> aIndices{perm.indexes[0], perm.indexes[2]};
    SmallVector<BlockArgument> bIndices{perm.indexes[2], perm.indexes[1]};
    SmallVector<int64_t> batches;

    // Generate the GEMM op; select inputs based on permutation order
    auto opC = cast<PxaReduceOp>(perm.values[0].getDefiningOp());
    auto opA = cast<PxaLoadOp>(perm.values[1].getDefiningOp());
    auto opB = cast<PxaLoadOp>(perm.values[2].getDefiningOp());

    // First, modify step size of all tiled indexes
    auto steps = op.getSteps();
    for (size_t i = 0; i < getBlockArgsAsSet().size(); i++) {
      bool foundBlockArg = false;
      for (size_t j = 0; j < getTiledIdxCount(); j++) {
        if (perm.indexes[j] == op.getBody()->getArgument(i)) {
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
      if (doBatch && !foundBlockArg && steps[i] == 1 &&
          getIdxRange(op.getBody()->getArgument(i)) > 1 &&
          isAdditionalReductionIndex(op.getBody()->getArgument(i),
                                     ArrayRef<Value>{opC, opA, opB})) {
        auto index = op.getBody()->getArgument(i);
        int64_t indexRange = getIdxRange(index);

        aIndices.emplace_back(index);
        bIndices.emplace_back(index);

        batches.emplace_back(indexRange);
        foundBlockArg = true;
        steps[i] = indexRange;
      }
    }
    op.setSteps(steps);

    // The first element of `batches` corresponds to the GEMM's 'k' index.
    // Additional reduction indices follow 'k'.
    batches.insert(batches.begin(), kBatches);

    OpBuilder builder = op.getBodyBuilder();

    SmallVector<Value> inputIndices, outputIndices;
    GemmOperand c(opC, {perm.indexes[0], perm.indexes[1]}, outputIndices);
    GemmOperand a(opA, aIndices, inputIndices);
    GemmOperand b(opB, bIndices, inputIndices);

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
        /*tile=*/builder.getI64ArrayAttr(tileSize),
        /*batches=*/builder.getI64ArrayAttr(batches),
        /*reductions=*/reductions);

    opC.result().replaceAllUsesWith(genericOp.getResult(0));
    opC.erase();
  }

public:
  StencilGEMM(AffineParallelOp op, unsigned numThreads, bool doBatch,
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
        numThreads{numThreads}, doBatch{doBatch}, stencilCostFn(costFn) {}
};

LogicalResult applyStencilGEMM(AffineParallelOp op, unsigned numThreads,
                               bool doBatch, StencilCostFunction costFn) {
  StencilGEMM stencil(op, numThreads, doBatch, costFn);
  stencil.DoStenciling();
  return success();
}

} // namespace pmlc::dialect::pxa
