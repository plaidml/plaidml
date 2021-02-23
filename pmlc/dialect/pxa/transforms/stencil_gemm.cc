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

  void computeBRGemmOffsets(
      const DenseMap<BlockArgument, SmallVector<int64_t, 8>> &offsets,
      SmallVector<int64_t, 8> &aOffsetsArray,
      SmallVector<int64_t, 8> &bOffsetsArray, int64_t numBatches) {
     
    for(size_t i = 0; i < (size_t) numBatches; i++){
        aOffsetsArray[i] = 0;
        bOffsetsArray[i] = 0;
    } 

    IVLOG(3, "numBatches in computeBRGEMM: " << numBatches);
    size_t innerStride = 1;
    
    for (auto &i : offsets) {
      int64_t aStride = i.second[0];
      int64_t bStride = i.second[1];
      int64_t indexRange = i.second[2];
      int64_t numSteps = i.second[3];

      IVLOG(3, "aStride in computeBRGEMM: " << aStride);           
      IVLOG(3, "bStride in computeBRGEMM: " << bStride);
      IVLOG(3, "indexRange in computeBRGEMM: " << indexRange);
      IVLOG(3, "numSteps in computeBRGEMM: " << numSteps);

      
      for (size_t k = 0; k < (size_t) numBatches; k += ((size_t)numSteps * innerStride)) {
        for (size_t j = 0; j < (size_t)numSteps; j++) {
          for (size_t l = 0; l < innerStride; l++) {
            aOffsetsArray[k + j * innerStride + l] += (j * (indexRange / numSteps) * aStride);
            bOffsetsArray[k + j * innerStride + l] += (j * (indexRange / numSteps) * bStride);
          }
        }
      }
      innerStride *= (size_t) numSteps;
    }
  }
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
    IVLOG(6,
          "Inner: loop = " << tot_inner_loop << " inner_time = " << inner_time);
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
      // If the index is from outside `op` this has already been logged, no
      // need for `else` branch here
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

    IVLOG(3, "Performance = " << perf << "(outer count: " << outer_batches
                              << ", middle count: " << tot_middle_loop
                              << ", startup cost: " << cost.startupCost
                              << ", inner time: " << inner_time << ")");
    return perf;
  }

  void transform(TensorAndIndexPermutation perm, ArrayRef<int64_t> tileSize) {
    int64_t numBatches = 1;
    int64_t kRange = getIdxRange(perm.indexes[2]);
    IVLOG(3, "kRange: " << kRange);
    int64_t kNumBatches = 1;
    // Generate the GEMM op; select inputs based on permutation order
    auto opC = cast<PxaReduceOp>(perm.values[0].getDefiningOp());
    auto opA = cast<PxaLoadOp>(perm.values[1].getDefiningOp());
    auto opB = cast<PxaLoadOp>(perm.values[2].getDefiningOp());
    DenseMap<BlockArgument, SmallVector<int64_t, 8> > offsets;
    auto bodyBuilder = op.getBodyBuilder();

    auto tileAttr = bodyBuilder.getI64ArrayAttr(tileSize);

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
            // We want to transform "regular" pxa.gemm where numBatches is 1:
            // affine.parallel (i, j, k) = (..., 0) to (..., kRange)
            //                             step (kStep) {
            //   pxa.gemm C[i, j] = A[i, k], B[k, j]: [..., kStep], 1
            // }
            //
            // to
            //
            // affine.parallel (i, j, k) = (..., 0) to (..., kRange) step (..,
            //                             step (kRange) {
            // pxa.gemm C[i, j] = A[i, k], B[k, j]: [..., kStep], (kRange/64)
            // }
            //
            // where the number of batches of A and B matrices to multiply is
            // the k loop's range divided by the original step size for the k
            // loop.
            //
            // Subsequently, kStep is set to kRange. That is, in one step, a
            // block of C is completely computed through reduction of batches
            // of A and B matrix multiplies.
            kNumBatches = (kRange / steps[i]);
            numBatches *= kNumBatches;
            steps[i] = kRange;
            // Insert into map to be used for offset-based batch reduce with
            // key = index and value = {stride, indexRange, numsteps}
            //

            auto aStrideInfo = getStrideInfo(opA);
            auto aStride = aStrideInfo->strides[perm.indexes[j]];
            auto bStrideInfo = getStrideInfo(opB);
            auto bStride = bStrideInfo->strides[perm.indexes[j]];

            auto aMemRefType = opA.getMemRef().getType().cast<MemRefType>();
            auto aElementType = aMemRefType.getElementType();
            aStride *= (aElementType.getIntOrFloatBitWidth()/ 8);

            auto bMemRefType = opB.getMemRef().getType().cast<MemRefType>();
            auto bElementType = bMemRefType.getElementType();
            bStride *= (bElementType.getIntOrFloatBitWidth()/ 8);


            offsets.insert(
                {perm.indexes[j],
                 SmallVector<int64_t, 8 >{aStride, bStride, kRange, kNumBatches}});

            IVLOG(3, "steps[" << i << "] = " << steps[i]);
          }
        }
      }
      if (doBatch && !foundBlockArg && steps[i] == 1 &&
          isAdditionalReductionIndex(op.getBody()->getArgument(i),
                                     ArrayRef<Value>{opC,
                                                     opA,
                                                     opB})) {
        
        
        auto index = op.getBody()->getArgument(i);
        // Compute stride of this index in A and B

        int64_t indexRange = getIdxRange(index);
        auto aStrideInfo = getStrideInfo(opA);
        auto aStride = aStrideInfo->strides[index];
        auto aMemRefType = opA.getMemRef().getType().cast<MemRefType>();
        auto aElementType = aMemRefType.getElementType();
        aStride *= (aElementType.getIntOrFloatBitWidth()/ 8); 


        auto bStrideInfo = getStrideInfo(opB);
        auto bStride = bStrideInfo->strides[index];

        auto bMemRefType = opB.getMemRef().getType().cast<MemRefType>();
        auto bElementType = bMemRefType.getElementType();
        bStride *= (bElementType.getIntOrFloatBitWidth()/ 8);
 



        // Insert into map to be used for offset-based batch reduce with key =
        // index and value = {stride, indexRange, numsteps}

        offsets.insert({index, SmallVector<int64_t, 8>{aStride, bStride, indexRange,
                                                 indexRange}});
        foundBlockArg = true;
        steps[i] = indexRange;
        IVLOG(3, "steps[" << i << "] = " << steps[i]);
        numBatches *= indexRange;
      }
    }
    op.setSteps(steps);
    IVLOG(3, "numBatches: " << numBatches);

    auto numBatchesAttr = bodyBuilder.getI64IntegerAttr(numBatches);

    // Iterate over offset map to generate offsets for offset-based brgemm
    SmallVector<int64_t, 8> aOffsetsArray(numBatches, 0), bOffsetsArray(numBatches, 0);
    computeBRGemmOffsets(offsets, aOffsetsArray, bOffsetsArray, numBatches);

    SmallVector<Value, 8> mapOperands;
    GemmOperand c(opC, {perm.indexes[0], perm.indexes[1]}, mapOperands);
    GemmOperand a(opA, {perm.indexes[0], perm.indexes[2]}, mapOperands);
    GemmOperand b(opB, {perm.indexes[2], perm.indexes[1]}, mapOperands);

    int64_t isContinuous = (doBatch && numBatches == kNumBatches) ? 1 : 0;
   
    auto brgemm = bodyBuilder.create<pxa::PxaGemmOp>(
        op.getLoc(), c.memref.getType(), //
        c.memref, AffineMapAttr::get(c.accessMap),
        AffineMapAttr::get(c.tileMap), //
        a.memref, AffineMapAttr::get(a.accessMap),
        AffineMapAttr::get(a.tileMap), //
        b.memref, AffineMapAttr::get(b.accessMap),
        AffineMapAttr::get(b.tileMap), //
        tileAttr, numBatchesAttr,
        bodyBuilder.getI64ArrayAttr(ArrayRef<int64_t>(aOffsetsArray)),
        bodyBuilder.getI64ArrayAttr(ArrayRef<int64_t>(bOffsetsArray)),
        bodyBuilder.getI64IntegerAttr(isContinuous),
        mapOperands);

    opC.result().replaceAllUsesWith(brgemm);
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
