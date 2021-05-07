// Copyright 2020, Intel Corporation

#include "pmlc/dialect/pxa/transforms/reorder_layouts.h"

#include <list>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "pmlc/dialect/pxa/analysis/affine_constraints.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/ir/interfaces.h"
#include "pmlc/dialect/pxa/ir/matchers.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/layout_utils.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/pxa/transforms/tile.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/matchers.h"
#include "pmlc/util/tags.h"

namespace pmlc::dialect::pxa {

class ReorderLayoutsPass final : public ReorderLayoutsBase<ReorderLayoutsPass> {
public:
  ReorderLayoutsPass() = default;
  explicit ReorderLayoutsPass(bool allowReorder, bool makeUserLayoutsExplicit) {
    this->allowReorder = allowReorder;
    this->makeUserLayoutsExplicit = makeUserLayoutsExplicit;
  }

  void runOnFunction() {
    mlir::FuncOp func = getFunction();
    mlir::DenseMap<mlir::Value, mlir::AffineMap> memLayoutMaps;
    llvm::SmallSet<mlir::AffineParallelOp, 4> parallelOps;

    if (recognizeConvsAndInsertBlockedDataLayouts) {
      recognizeConvsAndInsertBlockedDataLayouts(func, memLayoutMaps,
                                                parallelOps);
    }

    IVLOG(4, "Size of memLayoutMaps after Convolutions are recognized: "
                 << memLayoutMaps.size());

    mlir::DenseMap<mlir::Value, MemoryUsageDesc> globalMemory =
        gatherGlobalMemoryDescs(func, naiveScheduleModel);
    llvm::SetVector<mlir::Operation *> toRemove;

    for (auto &valueDesc : globalMemory) {
      MemoryUsageDesc &memoryDesc = valueDesc.second;
      IVLOG(3, "Optimizing layout for " << mlir::debugString(memoryDesc.value));
      mlir::Optional<ReorderDesc> optReorder = optimizeLayoutForReads(
          memoryDesc, memLayoutMaps, makeUserLayoutsExplicit);
      if (!optReorder.hasValue()) {
        IVLOG(3, "Could not select more optimal layout");
        continue;
      }
      ReorderDesc &reorder = optReorder.getValue();
      IVLOG(3, "Optimized layout: " << mlir::debugString(reorder.reorderMap));
      if (mlir::succeeded(convertMemoryLayout(memoryDesc.value, reorder))) {
        if (memoryDesc.parallelOp.hasValue()) {
          // parallelOps.insert(memoryDesc.parallelOp.getValue());
        }

        continue;
      }

      if (!allowReorder) {
        IVLOG(3,
              "Failed to change layout in-place, separate reorder not allowed");
        continue;
      }
      mlir::ModuleOp moduleOp = func->getParentOfType<mlir::ModuleOp>();

      IVLOG(3, "Failed to change layout in-place, inserting reorder");
      reorderMemoryReads(createReorder, reorder, memoryDesc, moduleOp,
                         toRemove);
      if (memoryDesc.parallelOp.hasValue()) {
        // parallelOps.insert(memoryDesc.parallelOp.getValue());
      }
    }

    static int count = 0;
    for (auto parallelOp : parallelOps) {
      /* if (count < 10) */ { tileLoopNestsToAlignWithDataMaps(parallelOp); }
      count++;
    }

    // Cleanup
    for (auto op : toRemove)
      op->erase();
  }
};

bool isPresent(llvm::SmallVector<mlir::Value, 4> resultOperands,
               mlir::Value arg) {
  for (size_t i = 0; i < resultOperands.size(); i++) {
    if (resultOperands[i] == arg) {
      return true;
    }
  }

  return false;
}

int intersectTwoSets(llvm::SmallVector<mlir::Value, 4> vec1,
                     llvm::SmallVector<mlir::Value, 4> vec2) {
  int common = 0;
  for (auto val : vec1) {
    if (isPresent(vec2, val)) {
      common++;
    }
  }

  return common;
}

void createBlockedLayoutForInputTensor(
    PxaLoadOp loadOp,
    mlir::DenseMap<mlir::Value, mlir::AffineMap> &memLayoutMaps) {
  mlir::Value indirectDef = getIndirectDef(loadOp.getMemRef());
  IVLOG(4, "indirectDef: " << mlir::debugString(indirectDef));

  auto srcMemType = loadOp.getMemRef().getType().cast<mlir::MemRefType>();
  mlir::ArrayRef<int64_t> shape = srcMemType.getShape();
  mlir::AffineMap map = loadOp.getAffineMap();
  mlir::MLIRContext *context = map.getContext();

  int64_t blockSize = 64;
  if (shape[3] % blockSize == 0) {
    //
    // *NHWC -> NCHW: newMap: (d0 d1 d2 d3) -> (d0 d3 d1 d2)
    // NCHW -> NCHWc16: newBlockedMap: (d0 d3 d1 d2) -> (d0 d3 floordiv 16, d1,
    // d2,d3 mod 16)
    //
    mlir::SmallVector<unsigned, 4> permutationMap;
    permutationMap.push_back(0);
    permutationMap.push_back(3);
    permutationMap.push_back(1);
    permutationMap.push_back(2);
    mlir::AffineMap newMap =
        mlir::AffineMap::getPermutationMap(permutationMap, context);
    IVLOG(4, "newMap: " << mlir::debugString(newMap));

    mlir::SmallVector<mlir::AffineExpr, 5> expansionExprs;
    for (unsigned idx = 0; idx < newMap.getNumResults(); ++idx) {
      mlir::AffineExpr expr;
      if (idx == 1) {
        expr = newMap.getResult(idx).floorDiv(blockSize);
      } else {
        expr = newMap.getResult(idx);
      }

      expansionExprs.push_back(expr);
      if (idx == newMap.getNumResults() - 1) {
        expansionExprs.push_back(newMap.getResult(1) % blockSize);
      }
    }

    mlir::AffineMap newBlockedMap = mlir::AffineMap::get(
        newMap.getNumResults(), 0, expansionExprs, context);
    IVLOG(4, "newBlockedMap: " << mlir::debugString(newBlockedMap));

    memLayoutMaps.insert({loadOp.getMemRef(), newBlockedMap});
  }
}

bool createBlockedLayoutForFilterTensor(
    PxaLoadOp loadOp,
    mlir::DenseMap<mlir::Value, mlir::AffineMap> &memLayoutMaps) {
  mlir::Value indirectDef = getIndirectDef(loadOp.getMemRef());
  IVLOG(4, "indirectDef for filter: " << mlir::debugString(indirectDef));

  auto srcMemType = loadOp.getMemRef().getType().cast<mlir::MemRefType>();
  mlir::ArrayRef<int64_t> shape = srcMemType.getShape();
  mlir::AffineMap map = loadOp.getAffineMap();
  mlir::MLIRContext *context = map.getContext();

  int64_t blockSize = 64;
  if (shape[2] % blockSize == 0 && shape[3] % blockSize == 0) {
    // RSCK -> C floordiv 16, K floordiv 16, R, S, C mod 16, K mod 16
    // RSCK -> CKRS (d0 d1 d2 d3) -> (d2 d3 d0 d1)
    // CKRS -> C floordiv 16, K floordiv 16, R, S, C mod 16, K mod 16
    // (d2 d3 d0 d1) -> (d2 floordiv 16, d3 floordiv 16, d0, d1, d2 mod 16, d3
    // mod 16)

    mlir::SmallVector<unsigned, 4> permutationMap;
    permutationMap.push_back(2);
    permutationMap.push_back(3);
    permutationMap.push_back(0);
    permutationMap.push_back(1);
    mlir::AffineMap newMap =
        mlir::AffineMap::getPermutationMap(permutationMap, context);
    IVLOG(4, "newMap: " << mlir::debugString(newMap));

    mlir::SmallVector<mlir::AffineExpr, 6> expansionExprs;
    for (unsigned idx = 0; idx < newMap.getNumResults(); ++idx) {
      mlir::AffineExpr expr;
      if (idx == 0 || idx == 1) {
        expr = newMap.getResult(idx).floorDiv(blockSize);
      } else {
        expr = newMap.getResult(idx);
      }

      expansionExprs.push_back(expr);
      if (idx == newMap.getNumResults() - 1) {
        expansionExprs.push_back(newMap.getResult(0) % blockSize);
        expansionExprs.push_back(newMap.getResult(1) % blockSize);
      }
    }

    mlir::AffineMap newBlockedMap = mlir::AffineMap::get(
        newMap.getNumResults(), 0, expansionExprs, context);
    IVLOG(4, "newBlockedMap: " << mlir::debugString(newBlockedMap));

    memLayoutMaps.insert({loadOp.getMemRef(), newBlockedMap});
    return true;
  }

  return false;
}

void recognizeConvsAndInsertBlockedDataLayouts(
    mlir::FuncOp func,
    mlir::DenseMap<mlir::Value, mlir::AffineMap> &memLayoutMaps,
    llvm::SmallSet<mlir::AffineParallelOp, 4> &parallelOps) {
  IVLOG(4, "Looking for Conv2ds");
  func.walk([&](mlir::AffineParallelOp parallelOp) {
    size_t numLoopsInConv2d = 7;
    IVLOG(4, "parallelOp.getSteps().size(): " << parallelOp.getSteps().size());
    if (parallelOp.getSteps().size() == numLoopsInConv2d) {
      IVLOG(4, "Found parallel loops that have " << numLoopsInConv2d);
      using mlir::matchers::m_Any;
      mlir::Value load1, load2, reduce;
      mlir::Operation *yield = parallelOp.getBody()->getTerminator();
      if (matchPattern(
              yield,
              mlir::m_Op<mlir::AffineYieldOp>(m_Capture(
                  &reduce,
                  m_PxaReduceOp(mlir::AtomicRMWKind::addf,
                                mlir::m_Op<mlir::MulFOp>(
                                    m_Capture(&load1, mlir::m_Op<PxaLoadOp>()),
                                    m_Capture(&load2, mlir::m_Op<PxaLoadOp>())),
                                m_Any()))))) {
        IVLOG(4, "Conv2d found");
        IVLOG(4, "Conv2dParallel Op: " << mlir::debugString(parallelOp));

        // Output - Input = %arg114 (the output channel)
        // Output - filter = %arg111, %arg112, %arg113 (NHW)

        IVLOG(4, "load1: " << mlir::debugString(load1));
        IVLOG(4, "load2: " << mlir::debugString(load2));
        IVLOG(4, "reduce: " << mlir::debugString(reduce));

        auto loadOp1 = mlir::dyn_cast<PxaLoadOp>(load1.getDefiningOp());
        auto loadOp2 = mlir::dyn_cast<PxaLoadOp>(load2.getDefiningOp());
        auto reduceOp = mlir::dyn_cast<PxaReduceOp>(reduce.getDefiningOp());

        unsigned expectedDimSizeOfTensors = 4;

        if (loadOp1 && loadOp2 && reduceOp &&
            loadOp1.getAffineMap().getNumResults() ==
                expectedDimSizeOfTensors &&
            loadOp2.getAffineMap().getNumResults() ==
                expectedDimSizeOfTensors &&
            reduceOp.getAffineMap().getNumResults() ==
                expectedDimSizeOfTensors) {
          llvm::SmallVector<mlir::Value, 4> loadOp1Operands;
          if (!getResultOperands(loadOp1.getAffineMap(), loadOp1.indices(),
                                 loadOp1Operands)) {
            return;
          }

          llvm::SmallVector<mlir::Value, 4> loadOp2Operands;
          if (!getResultOperands(loadOp2.getAffineMap(), loadOp2.indices(),
                                 loadOp2Operands)) {
            return;
          }

          llvm::SmallVector<mlir::Value, 4> reduceOperands;
          if (!getResultOperands(reduceOp.getAffineMap(), reduceOp.idxs(),
                                 reduceOperands)) {
            return;
          }

          int loadOp1ReduceCommon =
              intersectTwoSets(loadOp1Operands, reduceOperands);
          int loadOp2ReduceCommon =
              intersectTwoSets(loadOp2Operands, reduceOperands);

          IVLOG(4, "loadOp1ReduceCommon: " << loadOp1ReduceCommon);
          IVLOG(4, "loadOp2ReduceCommon: " << loadOp2ReduceCommon);

          int inputTensorCommon = 3, filterTensorCommon = 1;
          PxaLoadOp input, filter;
          if (loadOp1ReduceCommon == inputTensorCommon &&
              loadOp2ReduceCommon == filterTensorCommon) {
            input = loadOp1;
            filter = loadOp2;
          } else if (loadOp2ReduceCommon == inputTensorCommon &&
                     loadOp1ReduceCommon == filterTensorCommon) {
            input = loadOp2;
            filter = loadOp1;
          }

          static int count = 0;
          /* if (count < 1) */ {
            if (createBlockedLayoutForFilterTensor(filter, memLayoutMaps)) {
              createBlockedLayoutForInputTensor(input, memLayoutMaps);
            }
          }

          parallelOps.insert(parallelOp);
          count++;
          IVLOG(4, "count = " << count);
        }
      }
    }
  });
}

bool getResultOperands(mlir::AffineMap map, mlir::ValueRange mapOperands,
                       llvm::SmallVector<mlir::Value, 4> &resultOperands) {
  for (unsigned idx = 0; idx < map.getNumResults(); ++idx) {
    mlir::AffineExpr expr = map.getResult(idx);

    if (expr.getKind() == mlir::AffineExprKind::DimId) {
      auto dimExpr = expr.cast<mlir::AffineDimExpr>();
      unsigned pos = dimExpr.getPosition();
      IVLOG(4, "pos: " << pos);
      mlir::Value arg = mapOperands[pos];
      IVLOG(4, "arg: " << mlir::debugString(arg));
      if (!isPresent(resultOperands, arg)) {
        resultOperands.push_back(arg);
      }
    } else if (expr.getKind() == mlir::AffineExprKind::Add ||
               expr.getKind() == mlir::AffineExprKind::Mul) {
      auto addExpr = expr.cast<mlir::AffineBinaryOpExpr>();
      mlir::AffineExpr lhsExpr = addExpr.getLHS();
      mlir::AffineExpr rhsExpr = addExpr.getRHS();

      if (lhsExpr.getKind() == mlir::AffineExprKind::DimId) {
        auto dimExpr = lhsExpr.cast<mlir::AffineDimExpr>();
        unsigned pos = dimExpr.getPosition();
        auto arg = mapOperands[pos];
        if (!isPresent(resultOperands, arg)) {
          resultOperands.push_back(arg);
        }

      } else if (lhsExpr.getKind() == mlir::AffineExprKind::Constant) {
      } else {
        IVLOG(4, "Unhandled expression 1. Quitting "
                     << mlir::debugString(lhsExpr));
        return false;
      }

      if (rhsExpr.getKind() == mlir::AffineExprKind::DimId) {
        auto dimExpr = rhsExpr.cast<mlir::AffineDimExpr>();
        unsigned pos = dimExpr.getPosition();
        auto arg = mapOperands[pos];
        if (!isPresent(resultOperands, arg)) {
          resultOperands.push_back(arg);
        }

      } else if (rhsExpr.getKind() == mlir::AffineExprKind::Constant) {
      } else {
        IVLOG(4, "Unhandled expression 2. Quitting "
                     << mlir::debugString(rhsExpr));
        return false;
      }
    } else if (expr.getKind() == mlir::AffineExprKind::Constant) {
    } else {
      IVLOG(4, "Unhandled expression 3. Quitting " << mlir::debugString(expr));
      return false;
    }
  }

  return true;
}

bool divisorDividesTheLoops(int64_t constantValue,
                            mlir::AffineParallelOp outerParallelOp,
                            mlir::AffineParallelOp innerParallelOp,
                            size_t loopPos) {
  auto outerLoopLengths = outerParallelOp.getConstantRanges();
  auto innerLoopLengths = innerParallelOp.getConstantRanges();

  if (outerLoopLengths.hasValue() && innerLoopLengths.hasValue() &&
      outerLoopLengths.getValue().size() ==
          innerLoopLengths.getValue().size()) {
    int64_t outerLoopLength = outerLoopLengths.getValue()[loopPos];
    int64_t innerLoopLength = innerLoopLengths.getValue()[loopPos];
    IVLOG(4, "outerLoopLength: " << outerLoopLength
                                 << " innerLoopLength: " << innerLoopLength
                                 << " constantValue: " << constantValue);

    if (constantValue == innerLoopLength &&
        innerLoopLength <= outerLoopLength &&
        (outerLoopLength % innerLoopLength == 0)) {
      return true;
    }
  }

  return false;
}

bool isConstantRhsCompatibleWithSurroundingLoops(
    mlir::ValueRange innerIdxs, mlir::ValueRange outerIdxs,
    mlir::ValueRange mapOperands, mlir::AffineExpr &lhsExpr,
    mlir::AffineExpr &rhsExpr, size_t *innerLoopPos,
    mlir::AffineParallelOp outerParallelOp,
    mlir::AffineParallelOp innerParallelOp) {
  auto constantExpr = rhsExpr.cast<mlir::AffineConstantExpr>();
  int64_t constantValue = constantExpr.getValue();

  if (lhsExpr.getKind() == mlir::AffineExprKind::DimId) {
    auto dimExpr = lhsExpr.cast<mlir::AffineDimExpr>();
    unsigned pos = dimExpr.getPosition();
    IVLOG(4, "lhsExpr is DimId. Position " << pos);
    auto arg = mapOperands[pos];
    bool innerLoopPosFound = false;
    for (size_t i = 0; i < innerIdxs.size(); i++) {
      if (arg == innerIdxs[i]) {
        *innerLoopPos = i;
        innerLoopPosFound = true;
        IVLOG(4, "innerLoopPos: " << innerLoopPos);
        break;
      }
    }

    if (!innerLoopPosFound) {
      IVLOG(4, "innerLoopPos is not valid");
      return false;
    } else {
      if (*innerLoopPos < outerIdxs.size()) {
        if (divisorDividesTheLoops(constantValue, outerParallelOp,
                                   innerParallelOp, *innerLoopPos)) {
          return true;
        } else {
          IVLOG(4, "The divisor in the floordiv expression does not "
                   "divide the loop ranges.");
          return false;
        }
      } else {
        IVLOG(4, "innerLoopPos is out of bounds.");
        return false;
      }
    }

  } else {
    IVLOG(4, "The result expression is not a DimId kind expression.");
    return false;
  }

  return false;
}

struct MemRefSimplificationResults {
  bool newMapFormed;
  bool error;
  mlir::SmallVector<mlir::Value, 8> resultOperands;
  mlir::AffineMap simplifiedMap;
};

typedef struct MemRefSimplificationResults MemRefSimplificationResults;

MemRefSimplificationResults simplifyMemrefMapsInInnerLoops(
    mlir::AffineMap &map, mlir::ValueRange mapOperands,
    mlir::AffineParallelOp innerParallelOp,
    mlir::DenseMap<mlir::Value, mlir::Value> &varMap) {
  IVLOG(4, "Entered simplifyMemrefMapsInInnerLoops() CORE");

  MemRefSimplificationResults results;
  results.error = false;
  IVLOG(4, "map: " << mlir::debugString(map));

  unsigned currentNumDims = map.getNumDims();
  unsigned newDims = 0;
  results.newMapFormed = false;
  mlir::SmallVector<mlir::AffineExpr, 6> simplifiedExprs;

  for (unsigned i = 0; i < mapOperands.size(); i++) {
    results.resultOperands.push_back(mapOperands[i]);
  }

  for (unsigned idx = 0; idx < map.getNumResults(); ++idx) {
    mlir::AffineExpr expr = map.getResult(idx);
    bool expressionAdded = false;

    if (expr.getKind() == mlir::AffineExprKind::FloorDiv) {
      auto divExpr = expr.cast<mlir::AffineBinaryOpExpr>();
      mlir::AffineExpr lhsExpr = divExpr.getLHS();
      mlir::AffineExpr rhsExpr = divExpr.getRHS();

      if (rhsExpr.getKind() == mlir::AffineExprKind::Constant) {
        simplifiedExprs.push_back(lhsExpr);
        expressionAdded = true;
        results.newMapFormed = true;
      } else {
        IVLOG(4, "Error: The RHS of a floordiv is NOT a constant");
        results.error = true;
      }
    } else if (expr.getKind() == mlir::AffineExprKind::Mod) {
      auto modExpr = expr.cast<mlir::AffineBinaryOpExpr>();
      mlir::AffineExpr lhsExpr = modExpr.getLHS();
      mlir::AffineExpr rhsExpr = modExpr.getRHS();

      if (lhsExpr.getKind() == mlir::AffineExprKind::DimId &&
          rhsExpr.getKind() == mlir::AffineExprKind::Constant) {
        auto dimExpr = lhsExpr.cast<mlir::AffineDimExpr>();
        unsigned pos = dimExpr.getPosition();

        auto varMapIt = varMap.find(mapOperands[pos]);
        if (varMapIt == varMap.end()) {
          results.error = true;
          IVLOG(4, "Error: The map operand " << idx << " NOT found in varMap");
        } else {
          auto newDimIdExpr = mlir::getAffineDimExpr(currentNumDims + newDims,
                                                     map.getContext());
          newDims++;

          results.resultOperands.push_back(varMapIt->second);
          simplifiedExprs.push_back(newDimIdExpr);
          expressionAdded = true;
          results.newMapFormed = true;
        }
      } else {
        IVLOG(4, "Error: The RHS of a floordiv is NOT a constant");
        results.error = true;
      }
    }

    if (!expressionAdded) {
      simplifiedExprs.push_back(expr);
    }
  }

  if (results.error) {
    results.newMapFormed = false;
    return results;
  }

  if (results.newMapFormed) {
    results.simplifiedMap = mlir::AffineMap::get(
        map.getNumDims() + newDims, 0, simplifiedExprs, map.getContext());
    IVLOG(4, "resultOperands.size(): " << results.resultOperands.size());
    IVLOG(4, "simplifiedMap: " << mlir::debugString(results.simplifiedMap));
  }

  IVLOG(4, "Returned from simplifyMemrefMapsInInnerLoops() CORE");

  return results;
}

MemRefSimplificationResults scaleAndRewriteMemrefMapsInInnerLoops(
    mlir::AffineMap &map, mlir::ValueRange mapOperands,
    mlir::AffineParallelOp innerParallelOp,
    mlir::DenseMap<mlir::Value, mlir::Value> &varMap) {
  IVLOG(4, "Entered scaleAndRewriteMemrefMapsInInnerLoops()");

  auto innerLoopLengths = innerParallelOp.getConstantRanges();
  mlir::Block *body = innerParallelOp.getBody();
  auto innerIdxs = body->getArguments();

  MemRefSimplificationResults results;
  results.error = false;
  IVLOG(4, "map: " << mlir::debugString(map));

  unsigned currentNumDims = map.getNumDims();
  unsigned newDims = 0;
  results.newMapFormed = false;
  mlir::SmallVector<mlir::AffineExpr, 6> simplifiedExprs;

  for (unsigned i = 0; i < mapOperands.size(); i++) {
    results.resultOperands.push_back(mapOperands[i]);
  }

  for (unsigned idx = 0; idx < map.getNumResults(); ++idx) {
    bool expressionAdded = false;
    mlir::AffineExpr expr = map.getResult(idx);
    if (expr.getKind() == mlir::AffineExprKind::DimId) {
      auto dimExpr = expr.cast<mlir::AffineDimExpr>();
      unsigned pos = dimExpr.getPosition();

      auto varMapIt = varMap.find(mapOperands[pos]);
      if (varMapIt != varMap.end()) {
        auto newDimIdExpr =
            mlir::getAffineDimExpr(currentNumDims + newDims, map.getContext());
        newDims++;
        results.resultOperands.push_back(varMapIt->second);
        auto loopVar = varMapIt->second;
        int64_t multiplier = -1;
        for (size_t loopPos = 0; loopPos < innerIdxs.size(); loopPos++) {
          if (innerIdxs[loopPos] == loopVar) {
            multiplier = innerLoopLengths.getValue()[loopPos];
          }
        }

        if (multiplier == -1) {
          IVLOG(4, "The multiplier value couldn't be determined. Exiting");
          exit(1);
        }

        IVLOG(4, "Multiplier: " << multiplier);

        simplifiedExprs.push_back(expr * multiplier + newDimIdExpr);
        expressionAdded = true;
        results.newMapFormed = true;
      }
    }

    if (!expressionAdded) {
      simplifiedExprs.push_back(expr);
    }
  }

  if (results.error) {
    results.newMapFormed = false;
    return results;
  }

  if (results.newMapFormed) {
    results.simplifiedMap = mlir::AffineMap::get(
        map.getNumDims() + newDims, 0, simplifiedExprs, map.getContext());
    IVLOG(4, "resultOperands.size(): " << results.resultOperands.size());
    IVLOG(4, "simplifiedMap: " << mlir::debugString(results.simplifiedMap));
  }

  IVLOG(4, "Returning from scaleAndRewriteMemrefMapsInInnerLoops()");
  return results;
}

void simplifyMemrefMapsInInnerLoops(
    mlir::AffineParallelOp &parallelOp,
    mlir::DenseMap<mlir::Value, mlir::Value> &varMap) {
  IVLOG(4, "Entered simplifyMemrefMapsInInnerLoops()");

  parallelOp.walk([&](PxaLoadOp loadOp) {
    IVLOG(4, "PxaLoadOp: " << loadOp);

    mlir::AffineMap map = loadOp.getAffineMap();
    IVLOG(4, "map: " << mlir::debugString(map));

    mlir::OpBuilder builder(loadOp);
    MemRefSimplificationResults results = simplifyMemrefMapsInInnerLoops(
        map, loadOp.indices(), parallelOp, varMap);

    if (results.newMapFormed) {
      mlir::Value loadRes = builder.create<PxaLoadOp>(
          loadOp.getLoc(), loadOp.getMemRef(), results.simplifiedMap,
          /* loadOp.indices() */ results.resultOperands);
      loadOp.replaceAllUsesWith(loadRes);
      loadOp.erase();
    }
  });

  parallelOp.walk([&](PxaReduceOp reduceOp) {
    IVLOG(4, "PxaReduceOp: " << reduceOp);

    mlir::AffineMap map = reduceOp.getAffineMap();
    IVLOG(4, "map: " << mlir::debugString(map));

    mlir::OpBuilder builder(reduceOp);

    {
      MemRefSimplificationResults results = simplifyMemrefMapsInInnerLoops(
          map, reduceOp.idxs(), parallelOp, varMap);

      if (results.newMapFormed) {
        mlir::Value reduceRes = builder.create<PxaReduceOp>(
            reduceOp.getLoc(), reduceOp.getAgg(), reduceOp.val(),
            reduceOp.getMemRef(), results.simplifiedMap,
            results.resultOperands);

        reduceOp.replaceAllUsesWith(reduceRes);
        reduceOp.erase();
      }
    }

    {
      MemRefSimplificationResults results =
          scaleAndRewriteMemrefMapsInInnerLoops(map, reduceOp.idxs(),
                                                parallelOp, varMap);

      if (results.newMapFormed) {
        mlir::Value reduceRes = builder.create<PxaReduceOp>(
            reduceOp.getLoc(), reduceOp.getAgg(), reduceOp.val(),
            reduceOp.getMemRef(), results.simplifiedMap,
            results.resultOperands);

        reduceOp.replaceAllUsesWith(reduceRes);
        reduceOp.erase();
      }
    }
  });

  IVLOG(4, "Returning from simplifyMemrefMapsInInnerLoops()");
}

void tileLoopNestsToAlignWithDataMaps(mlir::AffineParallelOp &parallelOp) {
  mlir::DenseMap<mlir::Value, int64_t> tileSizeMap;
  bool tileSizesAreConsistent = true;
  IVLOG(4, "In tileLoopNestsToAlignWithDataMaps()");
  IVLOG(4, "parallelOp: " << mlir::debugString(parallelOp));

  mlir::Block *outerBody = parallelOp.getBody();
  auto outerIdxs = outerBody->getArguments();

  for (unsigned i = 0; i < outerIdxs.size(); ++i) {
    mlir::Value val = outerIdxs[i];
    IVLOG(4, "index i: " << i << ": " << mlir::debugString(val));
  }

  parallelOp.walk([&](PxaLoadOp op) {
    IVLOG(4, "read load op: " << op);
    mlir::Value memRef = op.getMemRef();
    IVLOG(4, "op.getMemRef(): " << mlir::debugString(memRef));
    IVLOG(4, "op.getMapOperands().size(): " << op.indices().size());

    mlir::AffineMap map = op.getAffineMap();
    IVLOG(4, "map: " << mlir::debugString(map));
    for (unsigned idx = 0; idx < map.getNumResults(); ++idx) {
      mlir::AffineExpr expr = map.getResult(idx);

      if (expr.getKind() == mlir::AffineExprKind::FloorDiv) {
        auto divExpr = expr.cast<mlir::AffineBinaryOpExpr>();
        mlir::AffineExpr rhsExpr = divExpr.getRHS();
        mlir::AffineExpr lhsExpr = divExpr.getLHS();

        if (rhsExpr.getKind() == mlir::AffineExprKind::Constant &&
            lhsExpr.getKind() == mlir::AffineExprKind::DimId) {
          auto constantExpr = rhsExpr.cast<mlir::AffineConstantExpr>();
          int64_t res = constantExpr.getValue();
          IVLOG(4, "The floor div constantValue: " << res);

          auto dimExpr = lhsExpr.cast<mlir::AffineDimExpr>();
          unsigned pos = dimExpr.getPosition();

          IVLOG(4, "lhsExpr-pos: " << pos);
          mlir::Value operand = op.indices()[pos];

          IVLOG(4, "operand: " << mlir::debugString(operand));

          for (unsigned i = 0; i < outerIdxs.size(); ++i) {
            mlir::Value loopVar = outerIdxs[i];

            if (loopVar == operand) {
              auto tileSizeMapIt = tileSizeMap.find(loopVar);
              if (tileSizeMapIt == tileSizeMap.end()) {
                IVLOG(4, "MATCH found. tile size = " << res);
                tileSizeMap.insert({loopVar, res});
              } else {
                int64_t existingTileSize = tileSizeMapIt->second;
                if (res != existingTileSize) {
                  IVLOG(4, "Tile Sizes are not consistent: res = "
                               << res << " old size: " << existingTileSize);
                  tileSizesAreConsistent = false;
                }
              }
            }
          }
        }
      }
    }

    IVLOG(4, "PxaLoadOp description ends");
  });

  for (unsigned i = 0; i < outerIdxs.size(); ++i) {
    mlir::Value val = outerIdxs[i];
    IVLOG(4, "index i: " << i << ": " << mlir::debugString(val));
  }

  if (tileSizesAreConsistent) {
    IVLOG(4, "Tile sizes are consistent. Performing tiling");
    mlir::SmallVector<int64_t, 6> tileSizes;
    bool nonUnitTileSizesPresent = false;
    for (unsigned i = 0; i < outerIdxs.size(); ++i) {
      mlir::Value loopVar = outerIdxs[i];
      auto tileSizeMapIt = tileSizeMap.find(loopVar);
      int64_t tileSize = 1;
      if (tileSizeMapIt != tileSizeMap.end()) {
        tileSize = tileSizeMapIt->second;

        if (tileSize != 1) {
          nonUnitTileSizesPresent = true;
        }
      }

      tileSizes.push_back(tileSize);
      IVLOG(4, "tile size: " << tileSize);
    }

    if (nonUnitTileSizesPresent) {
      // We will check that the tile sizes divide the loop lengths exactly.
      auto loopLengths = parallelOp.getConstantRanges();

      if (loopLengths.hasValue() &&
          loopLengths.getValue().size() == tileSizes.size()) {
        bool tileSizesDivideLoopLengths = true;
        for (size_t i = 0; i < loopLengths.getValue().size(); i++) {
          int64_t loopLength = loopLengths.getValue()[i];
          int64_t tileSize = tileSizes[i];
          IVLOG(4, "loopLength: " << loopLength << " tile size: " << tileSize);
          if (!(loopLength >= tileSize && loopLength % tileSize == 0)) {
            tileSizesDivideLoopLengths = false;
            break;
          }
        }

        if (tileSizesDivideLoopLengths) {
          mlir::AffineParallelOp innerLoops =
              performTiling(parallelOp, tileSizes);

          mlir::AffineMap outerLoopLowerMap =
              parallelOp.getLowerBoundsValueMap().getAffineMap();
          mlir::AffineMap outerLoopUpperMap =
              parallelOp.getUpperBoundsValueMap().getAffineMap();
          IVLOG(4,
                "outerLoopLowerMap: " << mlir::debugString(outerLoopLowerMap));
          IVLOG(4,
                "outerLoopUpperMap: " << mlir::debugString(outerLoopUpperMap));

          mlir::AffineValueMap lowerBoundsMap =
              innerLoops.getLowerBoundsValueMap();
          mlir::AffineMap lowerMap = lowerBoundsMap.getAffineMap();
          IVLOG(4, "lowerBoundsMap: " << mlir::debugString(lowerMap));

          mlir::AffineValueMap upperBoundsMap =
              innerLoops.getUpperBoundsValueMap();
          mlir::AffineMap upperMap = upperBoundsMap.getAffineMap();
          IVLOG(4, "upperBoundsMap: " << mlir::debugString(upperMap));

          unsigned numTileSizes = 0;
          // The following will set the lower bound and upper bound maps to
          // something like the following: Lower bounds: (d0, d1, d2 floordiv
          // 16, d3 floordiv 16, d4, d5) Upper bounds: (d0 + 1, d1 + 1, d2
          // floordiv 16 + 1, d3 floordiv 16 + 1, d4 + 1, d5 + 1)
          // TODO: the load/reduce ops' variables need to be modified too if
          // they depend upon the modified loop variables.
          /*
              %8 = affine.parallel (%arg0, %arg1, %arg2, %arg3, %arg4, %arg5,
            %arg6) = (0, 0, 0, 0, 0, 0, 0) to (1, 56, 56, 64, 3, 3, 64) step (1,
            1, 1, 16, 1, 1, 16) reduce ("assign") -> (memref<1x56x56x64xf32>) {
                %10 = affine.parallel (%arg7, %arg8, %arg9, %arg10, %arg11,
            %arg12, %arg13, %arg14, %arg15) = (%arg0, %arg1, %arg2, %arg3
            floordiv 16, %arg4, %arg5, %arg6 floordiv 16, 0, 0) to (%arg0 + 1,
            %arg1 + 1, %arg2 + 1, %arg3 floordiv 16 + 1, %arg4 + 1, %arg5 + 1,
            %arg6 floordiv 16 + 1, 16, 16) reduce ("assign") ->
            (memref<1x56x56x64xf32>) { %11 = pxa.load %4[%arg7, %arg13, %arg8 +
            %arg11, %arg9 + %arg12, %arg15] : memref<1x4x58x58x16xf32> %12 =
            pxa.load %6[%arg13, %arg10, %arg11, %arg12, %arg15, %arg14] :
            memref<4x4x3x3x16x16xf32> %13 = mulf %11, %12 : f32 %14 = pxa.reduce
            addf %13, %7[%arg7, %arg8, %arg9, %arg10] : memref<1x56x56x64xf32>
                  affine.yield %14 : memref<1x56x56x64xf32>
                }
                affine.yield %10 : memref<1x56x56x64xf32>
              }

            pxa.reduce addf %13, %7[%arg7, %arg8, %arg9, %arg10] ->
            pxa.reduce addf %13, %7[%arg7, %arg8, %arg9, %arg10 * 16 + %arg14]
          */
          for (size_t i = 0; i < tileSizes.size(); i++) {
            if (tileSizes[i] != 1) {
              lowerBoundsMap.setResult(
                  i, lowerBoundsMap.getResult(i).floorDiv(tileSizes[i]));
              upperBoundsMap.setResult(i, lowerBoundsMap.getResult(i) + 1);
              numTileSizes++;
            }
          }

          lowerMap = lowerBoundsMap.getAffineMap();
          IVLOG(4, "lowerBoundsMap: " << mlir::debugString(lowerMap));
          upperMap = upperBoundsMap.getAffineMap();
          IVLOG(4, "upperBoundsMap: " << mlir::debugString(upperMap));

          mlir::SmallVector<mlir::AffineExpr, 6> lowerExpandedExprs;
          mlir::SmallVector<mlir::AffineExpr, 6> upperExpandedExprs;
          unsigned currentNumDims = lowerMap.getNumResults();
          for (size_t i = 0; i < currentNumDims; i++) {
            lowerExpandedExprs.push_back(lowerMap.getResult(i));
            upperExpandedExprs.push_back(upperMap.getResult(i));
          }

          mlir::OpBuilder builder(lowerMap.getContext());
          for (size_t i = 0; i < tileSizes.size(); i++) {
            if (tileSizes[i] != 1) {
              auto lowerDimIdExpr = builder.getAffineConstantExpr(0);
              auto upperDimIdExpr = builder.getAffineConstantExpr(tileSizes[i]);
              lowerExpandedExprs.push_back(lowerDimIdExpr);
              upperExpandedExprs.push_back(upperDimIdExpr);
            }
          }

          mlir::AffineMap expandedLowerMap = mlir::AffineMap::get(
              currentNumDims, 0, lowerExpandedExprs, lowerMap.getContext());

          mlir::AffineMap expandedUpperMap = mlir::AffineMap::get(
              currentNumDims, 0, upperExpandedExprs, upperMap.getContext());

          IVLOG(4, "expandedLowerMap: " << mlir::debugString(expandedLowerMap));
          IVLOG(4, "expandedUpperMap: " << mlir::debugString(expandedUpperMap));

          innerLoops.setLowerBoundsMap(expandedLowerMap);
          innerLoops.setUpperBoundsMap(expandedUpperMap);

          llvm::SmallVector<int64_t, 8> steps = innerLoops.getSteps();
          for (size_t i = 0; i < numTileSizes; i++) {
            steps.push_back(1);
          }

          innerLoops.setSteps(steps);
          IVLOG(4, "The steps have been set.");

          int64_t numArguments = innerLoops.getBody()->getNumArguments();
          mlir::DenseMap<mlir::Value, mlir::Value> varMap;
          numTileSizes = 0;
          for (size_t i = 0; i < tileSizes.size(); i++) {
            if (tileSizes[i] != 1) {
              // FIXME: Create the type in a better fashion.
              if (numArguments > 0) {
                mlir::Type type =
                    innerLoops.getBody()->getArgument(0).getType();
                mlir::BlockArgument arg = innerLoops.getBody()->insertArgument(
                    numArguments + numTileSizes, type);
                numTileSizes++;
                varMap.insert({innerLoops.getBody()->getArgument(i), arg});
                IVLOG(4, "The new arg: " << mlir::debugString(arg));
              } else {
                IVLOG(4, "Error. Exiting");
                exit(1);
              }
            }
          }

          // TODO: Establish the conditions under which simplifying the affine
          // expressions is OK
          simplifyMemrefMapsInInnerLoops(innerLoops, varMap);
        }
      }
    }
  }

  IVLOG(4, "modified_parallelOp: " << mlir::debugString(parallelOp));
}
// =============================================================================
// gatherGlobalMemoryDescs - helpers and implementation.
// =============================================================================
/// Based on ScheduleModel model selects operands order from biggest to
/// lowest cost.
static void calculateSchedule(mlir::Operation *op, mlir::ValueRange operands,
                              const ScheduleModel &model,
                              mlir::SmallVectorImpl<unsigned> &schedule) {
  struct DimDesc {
    unsigned level;
    unsigned operand;
  };

  std::list<mlir::AffineParallelOp> parallelNest;
  mlir::Operation *parallel = op;
  while (auto next = parallel->getParentOfType<mlir::AffineParallelOp>()) {
    parallelNest.push_front(next);
    parallel = next.getOperation();
  }

  std::vector<DimDesc> descs;
  for (mlir::Value operand : operands) {
    auto arg = operand.dyn_cast<mlir::BlockArgument>();
    mlir::Operation *parent = arg.getOwner()->getParentOp();
    unsigned idx = 0;
    auto it = parallelNest.begin();
    while (it->getOperation() != parent)
      idx++, it++;
    descs.push_back(DimDesc{idx, arg.getArgNumber()});
  }
  LoopNestSchedule info = model(std::vector<mlir::AffineParallelOp>(
      parallelNest.begin(), parallelNest.end()));
  for (unsigned idx = 0; idx < operands.size(); ++idx)
    schedule.push_back(idx);
  std::stable_sort(schedule.begin(), schedule.end(),
                   [&](unsigned a, unsigned b) {
                     const DimDesc &descA = descs[a];
                     const DimDesc &descB = descs[b];
                     return info[descA.level][descA.operand] >
                            info[descB.level][descB.operand];
                   });
}

/// Gathers information about specified read operation.
static MemoryReadDesc gatherReadDesc(PxaReadOpInterface op,
                                     const ScheduleModel &scheduleModel) {
  mlir::MemRefType memRefType = op.getMemRefType();
  mlir::ArrayRef<int64_t> shapeRef = memRefType.getShape();
  mlir::SmallVector<int64_t, 4> readVec(shapeRef.size(), 1);
  if (auto vecRead = mlir::dyn_cast<PxaVectorLoadOp>(op.getOperation())) {
    auto vecType = vecRead.getType().cast<mlir::VectorType>();
    mlir::ArrayRef<int64_t> vecShape = vecType.getShape();
    for (unsigned idx = 0; idx < vecShape.size(); ++idx)
      readVec[readVec.size() - vecShape.size() + idx] = vecShape[idx];
  }
  mlir::AffineMap readMap = op.getAffineMap();
  mlir::Operation::operand_range mapOperands = op.getMapOperands();
  mlir::FlatAffineConstraints dimensionConstraints =
      gatherAffineMapConstraints(mlir::AffineValueMap(readMap, mapOperands));
  mlir::SmallVector<unsigned, 6> iterationOrder;
  calculateSchedule(op.getOperation(), mapOperands, scheduleModel,
                    iterationOrder);

  return MemoryReadDesc{op, op.getAffineMap(), std::move(readVec),
                        std::move(dimensionConstraints),
                        std::move(iterationOrder)};
}

/// Gathers information about specified write operation.
static MemoryWriteDesc gatherWriteDesc(PxaReduceOpInterface op,
                                       mlir::AffineParallelOp parallelOp) {
  mlir::MemRefType memRefType = op.getMemRefType();
  mlir::ArrayRef<int64_t> shapeRef = memRefType.getShape();
  mlir::SmallVector<int64_t, 4> reduceVec(shapeRef.size(), 1);
  if (auto vecReduce = mlir::dyn_cast<PxaVectorReduceOp>(op.getOperation())) {
    auto vecType = vecReduce.getVectorType();
    mlir::ArrayRef<int64_t> vecShape = vecType.getShape();
    for (unsigned idx = 0; idx < vecShape.size(); ++idx)
      reduceVec[reduceVec.size() - vecShape.size() + idx] = vecShape[idx];
  }
  return MemoryWriteDesc{op, parallelOp, std::move(reduceVec)};
}

/// Returns MemoryUsageDesc initialized with information about `memory`,
/// without any information about its usage.
static MemoryUsageDesc getEmptyUsageDesc(mlir::Value memory) {
  auto memoryType = memory.getType().cast<mlir::MemRefType>();
  mlir::ArrayRef<int64_t> shapeRef = memoryType.getShape();
  mlir::SmallVector<int64_t, 4> shape(shapeRef.begin(), shapeRef.end());
  auto desc = MemoryUsageDesc{memory, shape, llvm::None};
  desc.count = std::accumulate(shapeRef.begin(), shapeRef.end(),
                               /*init=*/(int64_t)1, std::multiplies<int64_t>());
  return desc;
}

mlir::DenseMap<mlir::Value, MemoryUsageDesc>
gatherGlobalMemoryDescs(mlir::FuncOp func, const ScheduleModel &model) {
  mlir::DenseMap<mlir::Value, MemoryUsageDesc> globalMemory;

  auto getOrCreateGlobalDesc = [&](mlir::Value memory) -> MemoryUsageDesc & {
    auto memoryIt = globalMemory.find(memory);
    if (memoryIt == globalMemory.end()) {
      MemoryUsageDesc memoryDesc = getEmptyUsageDesc(memory);
      memoryIt = globalMemory.insert({memory, memoryDesc}).first;
    }
    return memoryIt->second;
  };

  for (auto parallelOp : func.getOps<mlir::AffineParallelOp>()) {
    parallelOp.walk([&](PxaReadOpInterface read) {
      mlir::Value indirectDef = getIndirectDef(read.getMemRef());
      // Skip memory local to `affine.parallel`.
      if (!parallelOp.isDefinedOutsideOfLoop(indirectDef))
        return;
      MemoryUsageDesc &memoryDesc = getOrCreateGlobalDesc(indirectDef);
      memoryDesc.reads.emplace_back(gatherReadDesc(read, model));
      memoryDesc.parallelOp = parallelOp;
    });
    parallelOp.walk([&](PxaReduceOpInterface reduce) {
      mlir::Value indirectDef = getIndirectDef(reduce.getMemRef());
      // Skip memory local to `affine.parallel`.
      if (!parallelOp.isDefinedOutsideOfLoop(indirectDef))
        return;
      MemoryUsageDesc &memoryDesc = getOrCreateGlobalDesc(indirectDef);
      memoryDesc.writes.emplace_back(gatherWriteDesc(reduce, parallelOp));
      memoryDesc.parallelOp = parallelOp;
    });
  }

  return globalMemory;
}

// =============================================================================
// optimizeLayoutForReads - helpers and implementation.
// =============================================================================
/// Walks over all read and write descriptions and selects common non-unit
/// vectorization and stores its reference to "result".
/// If all accesses are not vectorized then stores unit vectorization.
/// Returns mlir::failure() if there is more than one non-unit vectorizations.
static mlir::LogicalResult
selectCommonVectorization(MemoryUsageDesc &memoryDesc,
                          mlir::ArrayRef<int64_t> &result) {
  bool isResultUnit = false;
  auto isUnitVector = [](mlir::ArrayRef<int64_t> vec) {
    return std::all_of(vec.begin(), vec.end(),
                       [](int64_t val) { return val == 1; });
  };

  for (MemoryReadDesc &readDesc : memoryDesc.reads) {
    mlir::ArrayRef<int64_t> readVector = readDesc.readVector;
    if (result.empty() || isResultUnit) {
      result = readVector;
      isResultUnit = isUnitVector(readVector);
      continue;
    }
    if (isUnitVector(readVector))
      continue;
    if (!std::equal(result.begin(), result.end(), readVector.begin()))
      return mlir::failure();
  }
  for (MemoryWriteDesc &writeDesc : memoryDesc.writes) {
    mlir::ArrayRef<int64_t> writeVector = writeDesc.writeVector;
    if (result.empty() || isResultUnit) {
      result = writeVector;
      isResultUnit = isUnitVector(writeVector);
      continue;
    }
    if (isUnitVector(writeVector))
      continue;
    if (!std::equal(result.begin(), result.end(), writeVector.begin()))
      return mlir::failure();
  }
  return mlir::success();
}

void fixResultsIfModulosInAffineMap(
    mlir::AffineMap &map, mlir::SmallVector<int64_t, 6> &expandedShape) {
  // Deal with modulos
  mlir::SmallVector<int64_t, 6> expandedShapeCopy(expandedShape);
  expandedShape.clear();

  IVLOG(3, "map.getNumResults(): " << map.getNumResults());
  for (unsigned idx = 0; idx < map.getNumResults(); ++idx) {
    int64_t res = expandedShapeCopy[idx];

    mlir::AffineExpr expr = map.getResult(idx);
    IVLOG(3, "dimExpr: " << mlir::debugString(expr));

    if (expr.getKind() == mlir::AffineExprKind::Mod) {
      auto addExpr = expr.cast<mlir::AffineBinaryOpExpr>();
      mlir::AffineExpr lhsExpr = addExpr.getLHS();
      mlir::AffineExpr rhsExpr = addExpr.getRHS();

      IVLOG(3, "Modulo Op, LHS: " << mlir::debugString(lhsExpr));
      IVLOG(3, "Modulo Op, RHS: " << mlir::debugString(rhsExpr));

      if (rhsExpr.getKind() == mlir::AffineExprKind::Constant) {
        auto constantExpr = rhsExpr.cast<mlir::AffineConstantExpr>();
        res = constantExpr.getValue();
        IVLOG(3, "The modulo constantValue: " << res);
      }
    }

    expandedShape.push_back(res);
  }
}

mlir::LogicalResult
applyMapOnConstantArray(mlir::AffineMap map, mlir::ArrayRef<int64_t> &input,
                        mlir::SmallVector<int64_t, 6> &expandedShape) {
  mlir::SmallVector<mlir::AffineExpr, 6> expansionExprs;
  // mlir::SmallVector<int64_t, 6> expandedShape;
  mlir::SmallVector<int64_t, 6> expandedVec;
  IVLOG(3, "applyMapOnConstantArray map: " << mlir::debugString(map));

  mlir::SmallVector<mlir::Attribute, 8> operandConstants;
  mlir::OpBuilder builder(map.getContext());

  for (auto i : input) {
    auto attr = builder.getI64IntegerAttr(i);
    operandConstants.push_back(attr);
  }

  mlir::SmallVector<mlir::Attribute, 4> foldedResults;
  if (mlir::failed(map.constantFold(operandConstants, foldedResults))) {
    return mlir::failure();
  } else {
    for (unsigned i = 0; i < foldedResults.size(); i++) {
      int64_t val =
          foldedResults[i].cast<mlir::IntegerAttr>().getValue().getSExtValue();
      expandedShape.push_back(val);
    }

    fixResultsIfModulosInAffineMap(map, expandedShape);
    IVLOG(3, "RESULTS: ");
    for (unsigned i = 0; i < expandedShape.size(); i++) {
      IVLOG(3, "val: " << expandedShape[i]);
    }

    return mlir::success();
  }
}

mlir::Optional<ReorderDesc> chooseUserProvidedTargetLayout(
    MemoryUsageDesc &memoryDesc,
    mlir::DenseMap<mlir::Value, mlir::AffineMap> &memLayoutMaps) {
  IVLOG(3, "In chooseUserProvidedTargetLayout()\n");

  mlir::Optional<ReorderDesc> selectedReorder = llvm::None;

  if (memoryDesc.reads.size() > 0) {
    MemoryReadDesc &readDesc = memoryDesc.reads.front();

    PxaReadOpInterface readOp = readDesc.readOp;
    mlir::Value readMem = readOp.getMemRef();
    IVLOG(3, "readMem: " << mlir::debugString(readMem));

    mlir::MemRefType memrefType = readOp.getMemRefType();

    bool layoutSet = false;
    mlir::AffineMap layoutMap;

    // First we check if a layout is set with the memref itself
    if (memrefType.getAffineMaps().size() > 0) {
      layoutMap = memrefType.getAffineMaps().front();
      IVLOG(3, "layoutMap: " << mlir::debugString(layoutMap));
      layoutSet = true;
    }

    // Then we check if a layout is set by any of the pattern recognizers
    // such as convolution recognizers
    if (!layoutSet) {
      auto memLayoutMapsIt = memLayoutMaps.find(readMem);
      if (memLayoutMapsIt != memLayoutMaps.end()) {
        layoutMap = memLayoutMapsIt->second;
        layoutSet = true;
        IVLOG(4, "layoutMap for conv2d: " << mlir::debugString(layoutMap));
      }
    }

    if (layoutSet) {
      mlir::ArrayRef<int64_t> tensorShape = memrefType.getShape();
      IVLOG(3, "Extant shape: ");

      for (auto i : tensorShape) {
        IVLOG(3, "i: " << i);
      }

      mlir::SmallVector<int64_t, 6> expandedShape;
      mlir::SmallVector<int64_t, 6> expandedVec;

      if (mlir::succeeded(
              applyMapOnConstantArray(layoutMap, tensorShape, expandedShape))) {
        for (size_t i = 0; i < expandedShape.size(); i++) {
          expandedVec.push_back(1);
        }

        selectedReorder = ReorderDesc{layoutMap, expandedShape, expandedVec};
      }
    }
  }
  return selectedReorder;
}

void printSmallVector(mlir::ArrayRef<int64_t> vec) {
  IVLOG(3, "Vector: ");
  for (int64_t i : vec) {
    IVLOG(3, " " << i);
  }

  IVLOG(3, "\n");
}

mlir::Optional<ReorderDesc>
optimizeLayoutForReads(MemoryUsageDesc &memoryDesc,
                       bool makeUserLayoutsExplicit) {
  mlir::DenseMap<mlir::Value, mlir::AffineMap> memLayoutMaps;
  return optimizeLayoutForReads(memoryDesc, memLayoutMaps,
                                makeUserLayoutsExplicit);
}

mlir::Optional<ReorderDesc> optimizeLayoutForReads(
    MemoryUsageDesc &memoryDesc,
    mlir::DenseMap<mlir::Value, mlir::AffineMap> &memLayoutMaps,
    bool makeUserLayoutsExplicit) {
  mlir::Optional<ReorderDesc> selectedReorder = llvm::None;

  if (makeUserLayoutsExplicit) {
    // FIXME: short circuiting other reordering logic
    selectedReorder = chooseUserProvidedTargetLayout(memoryDesc, memLayoutMaps);
    return selectedReorder;
  }

  if (!selectedReorder.hasValue()) {
    mlir::ArrayRef<int64_t> commonVector;
    if (mlir::failed(selectCommonVectorization(memoryDesc, commonVector))) {
      IVLOG(3, "Inconsistent vectorization between reads and writes");
      return llvm::None;
    }
    for (MemoryReadDesc &readDesc : memoryDesc.reads) {
      IVLOG(3, "readDesc.readMap: " << mlir::debugString(readDesc.readMap));
      mlir::Optional<ReorderDesc> reorder =
          tileAffineMap(readDesc.readMap, memoryDesc.shape, commonVector,
                        readDesc.dimensionConstraints, readDesc.iterationOrder);
      if (!reorder.hasValue())
        return llvm::None;
      if (!selectedReorder.hasValue()) {
        selectedReorder = reorder;

        IVLOG(3,
              "reorderMap: " << mlir::debugString(selectedReorder->reorderMap));
        IVLOG(3, "reorderedShape: ");
        printSmallVector(selectedReorder->reorderedShape);
        IVLOG(3, "reorderedVector: ");
        printSmallVector(selectedReorder->reorderedVector);

        continue;
      }
      if (selectedReorder->reorderMap != reorder->reorderMap) {
        IVLOG(3, "Inconsistent layout between reads");
        return llvm::None;
      }
    }
  } else {
    IVLOG(3, "The user specified layout has been used.");
    IVLOG(3, "reorderMap: " << mlir::debugString(selectedReorder->reorderMap));
    IVLOG(3, "reorderedShape: ");
    printSmallVector(selectedReorder->reorderedShape);
    IVLOG(3, "reorderedVector: ");
    printSmallVector(selectedReorder->reorderedVector);
  }

  return selectedReorder;
}

// =============================================================================
// naiveScheduleModel - implementation.
// =============================================================================
LoopNestSchedule
naiveScheduleModel(mlir::ArrayRef<mlir::AffineParallelOp> loopNest) {
  LoopNestSchedule result;
  result.resize(loopNest.size());
  int64_t scheduleAcc = 1;
  // Process loops in reverse order, from inner most to outer most.
  for (unsigned level = loopNest.size(); level > 0; --level) {
    mlir::AffineParallelOp parallel = loopNest[level - 1];
    OperandSchedule &operandSched = result[level - 1];
    operandSched.resize(parallel.getIVs().size());
    // Process operands in reverse order.
    for (unsigned argIdx = operandSched.size(); argIdx > 0; --argIdx) {
      operandSched[argIdx - 1] = scheduleAcc;
      scheduleAcc += 1;
    }
  }
  return result;
}

// ============================================================================
// Helper function to get pack\unpack ops.
// ============================================================================
template <typename OpType>
void getPackOp(OpType &packOp, mlir::FuncOp funcOp) {
  // Assume there is single pack op in function
  auto packOps = funcOp.getOps<OpType>();
  if (!packOps.empty())
    packOp = *packOps.begin();
}

// ============================================================================
// Helper function to replace unpackOps with updated types in sync with packOp.
// ============================================================================
pmlc::dialect::stdx::UnpackOp
updateUnpackOp(pmlc::dialect::stdx::UnpackOp unpackOp,
               pmlc::dialect::stdx::PackOp packOp,
               llvm::SetVector<mlir::Operation *> &toRemove) {
  mlir::OpBuilder builder(unpackOp);
  auto newUnpackOp = builder.create<pmlc::dialect::stdx::UnpackOp>(
      unpackOp.getLoc(), packOp.getOperandTypes(), unpackOp.in());
  unpackOp.replaceAllUsesWith(newUnpackOp);
  if (!toRemove.count(unpackOp))
    toRemove.insert(unpackOp);
  return newUnpackOp;
}

// =============================================================================
// naiveScheduleModel - implementation.
// =============================================================================
void reorderMemoryReads(const ReorderCreator &creator, ReorderDesc &reorderDesc,
                        MemoryUsageDesc &memoryDesc, mlir::ModuleOp &moduleOp,
                        llvm::SetVector<mlir::Operation *> &toRemove) {
  mlir::DenseSet<mlir::Value> memoryToReorder;
  for (MemoryReadDesc &readDesc : memoryDesc.reads) {
    PxaReadOpInterface readOp = readDesc.readOp;
    mlir::Value readMem = readOp.getMemRef();
    memoryToReorder.insert(readMem);
  }

  // Check for init and main functions for pack and unpack ops,
  // assume there is single pack and unpack invocation
  pmlc::dialect::stdx::PackOp packOp;
  pmlc::dialect::stdx::UnpackOp mainUnpackOp, finiUnpackOp;
  if (moduleOp) {
    auto initFunc = moduleOp.lookupSymbol<mlir::FuncOp>("init");
    auto mainFunc = moduleOp.lookupSymbol<mlir::FuncOp>("main");
    auto finiFunc = moduleOp.lookupSymbol<mlir::FuncOp>("fini");
    if (mainFunc && initFunc && finiFunc) {
      getPackOp(packOp, initFunc);
      getPackOp(mainUnpackOp, mainFunc);
      getPackOp(finiUnpackOp, finiFunc);
    }
  }

  for (mlir::Value originalMem : memoryToReorder) {
    mlir::OpBuilder builder(originalMem.getContext());
    builder.setInsertionPointAfterValue(originalMem);

    auto memToReorder = originalMem;
    auto loc = originalMem.getLoc();
    auto unpackIdx = 0;
    // Check if memory comes from init function, if so, create new reorder there
    if (originalMem.getDefiningOp()) {
      // Get unpack op to know if data comes from init
      if (mlir::isa<pmlc::dialect::stdx::UnpackOp>(
              originalMem.getDefiningOp())) {
        if (auto unpackAsResult = originalMem.dyn_cast<mlir::OpResult>()) {
          // Get index of the buffer so we could map it later in init
          unpackIdx = unpackAsResult.getResultNumber();
          // Replace originalMem with the pack op operand
          memToReorder = packOp.getOperand(unpackIdx);
          // Move the new function insert point to init
          builder.setInsertionPoint(packOp);
          loc = packOp.getLoc();
        }
      }
    }

    // TODO: It should be fused location of all reads.
    // Create the data copy operation after all the Writes
    if (memoryDesc.writes.size() > 0) {
      builder.setInsertionPointAfter(
          memoryDesc.writes[memoryDesc.writes.size() - 1]
              .surroundingParallelOp);
    }

    mlir::Value reorderedMem = creator(loc, builder, reorderDesc, memToReorder);
    replaceMemoryLayoutForReading(reorderedMem, memToReorder, reorderDesc);

    if (memToReorder != originalMem) {
      // Update the pack operand with new reordered mem
      packOp.setOperand(unpackIdx, reorderedMem);

      // Update the unpack functions in both main and fini
      // TODO: move this part outside of the loop or create option to update the
      // result type in the unpack ops so we would not need to replace
      // the op per mem reorder
      auto newMainUnpackOp = updateUnpackOp(mainUnpackOp, packOp, toRemove);
      replaceMemoryLayoutForReading(newMainUnpackOp.getResult(unpackIdx),
                                    originalMem, reorderDesc);
      auto newFiniUnpackOp = updateUnpackOp(finiUnpackOp, packOp, toRemove);
      replaceMemoryLayoutForReading(newFiniUnpackOp.getResult(unpackIdx),
                                    originalMem, reorderDesc);
    }
  }
}

// ============================================================================
// Helper affine map transformations
// ============================================================================
static void
expandAffineExpr(mlir::AffineExpr expr, mlir::AffineExpr dimExpr,
                 int64_t dimSize, int64_t vecSize,
                 mlir::FlatAffineConstraints &constraints,
                 mlir::SmallVectorImpl<mlir::AffineExpr> &expansionExprs,
                 mlir::SmallVectorImpl<int64_t> &expandedShape,
                 mlir::SmallVectorImpl<int64_t> &expandedVec) {
  auto ceilDiv = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
  if (vecSize != 1) {
    expandAffineExpr(expr.floorDiv(vecSize), dimExpr.floorDiv(vecSize),
                     ceilDiv(dimSize, vecSize), 1, constraints, expansionExprs,
                     expandedShape, expandedVec);
    expansionExprs.push_back(dimExpr % vecSize);
    expandedShape.push_back(vecSize);
    expandedVec.push_back(vecSize);
    return;
  }
  if (expr.getKind() == mlir::AffineExprKind::Add) {
    auto addExpr = expr.cast<mlir::AffineBinaryOpExpr>();
    mlir::AffineExpr lhsExpr = addExpr.getLHS();
    mlir::AffineExpr rhsExpr = addExpr.getRHS();
    mlir::Optional<int64_t> lhsUpperBound = getUpperBound(lhsExpr, constraints);
    mlir::Optional<int64_t> rhsUpperBound = getUpperBound(rhsExpr, constraints);

    // Pattern e*i* + e*j*, where e*i* % N == 0 and e*j* < N.
    mlir::Optional<bool> caseRhsSmaller = rhsUpperBound.map(
        [&](int64_t val) { return lhsExpr.isMultipleOf(val + 1); });
    // Pattern e*i* + e*j*, where e*i* < N and e*j* % N == 0.
    mlir::Optional<bool> caseLhsSmaller = lhsUpperBound.map(
        [&](int64_t val) { return rhsExpr.isMultipleOf(val + 1); });

    if (caseRhsSmaller.getValueOr(false)) {
      int64_t divisor = rhsUpperBound.getValue() + 1;
      expandAffineExpr(lhsExpr.floorDiv(divisor), dimExpr.floorDiv(divisor),
                       ceilDiv(dimSize, divisor), vecSize, constraints,
                       expansionExprs, expandedShape, expandedVec);
      expandAffineExpr(rhsExpr, dimExpr % divisor, divisor, vecSize,
                       constraints, expansionExprs, expandedShape, expandedVec);
      return;
    }
    if (caseLhsSmaller.getValueOr(false)) {
      int64_t divisor = lhsUpperBound.getValue() + 1;
      expandAffineExpr(rhsExpr.floorDiv(divisor), dimExpr.floorDiv(divisor),
                       ceilDiv(dimSize, divisor), vecSize, constraints,
                       expansionExprs, expandedShape, expandedVec);
      expandAffineExpr(lhsExpr, dimExpr % divisor, divisor, vecSize,
                       constraints, expansionExprs, expandedShape, expandedVec);
      return;
    }
  } else if (expr.getKind() == mlir::AffineExprKind::FloorDiv) {
    IVLOG(3, "FLOORDIV: ");
  }

  expansionExprs.push_back(dimExpr);
  expandedShape.push_back(dimSize);
  expandedVec.push_back(vecSize);
}

ReorderDesc expandAffineMap(mlir::AffineMap map, mlir::ArrayRef<int64_t> shape,
                            mlir::ArrayRef<int64_t> vector,
                            mlir::FlatAffineConstraints &constraints) {
  IVLOG(3, "Input map: " << mlir::debugString(map));
  mlir::SmallVector<mlir::AffineExpr, 6> expansionExprs;
  mlir::SmallVector<int64_t, 6> expandedShape;
  mlir::SmallVector<int64_t, 6> expandedVec;
  for (unsigned idx = 0; idx < map.getNumResults(); ++idx) {
    mlir::AffineExpr dimExpr = mlir::getAffineDimExpr(idx, map.getContext());
    expandAffineExpr(map.getResult(idx), dimExpr, shape[idx], vector[idx],
                     constraints, expansionExprs, expandedShape, expandedVec);
  }
  auto reorderMap = mlir::AffineMap::get(map.getNumResults(), 0, expansionExprs,
                                         map.getContext());

  IVLOG(3, "Output map: " << mlir::debugString(reorderMap));
  return ReorderDesc{reorderMap, expandedShape, expandedVec};
}

ReorderDesc sortAffineMap(mlir::AffineMap map, mlir::ArrayRef<int64_t> shape,
                          mlir::ArrayRef<int64_t> vector,
                          mlir::ArrayRef<unsigned> schedule) {
  if (map.getNumInputs() > 31) {
    // TODO: Add support for larger number of dimensions.
    mlir::emitWarning(mlir::UnknownLoc::get(map.getContext()),
                      "sorting affine map unsupported (> 31 inputs)")
            .attachNote()
        << "see affine map: " << mlir::debugString(map);
    auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(
        map.getNumDims(), map.getContext());
    return ReorderDesc{identityMap,
                       {shape.begin(), shape.end()},
                       {vector.begin(), vector.end()}};
  }
  // Small trick with order induced by norm for sorting.
  // For schedule <s0, s1, s2, .., s*n*>, each expression can be thought
  // as boolean vector, where i-th coordinate signifies wheter expression uses
  // i-th dimension from schedule.
  //
  // To transform such vector into norm with desired properties follwoing can
  // be used:
  // 1. Reverse values to the left of rightmost "1", ie:
  //    <a, b, c, 1, 0...> -> <c, b, a, 1, 0...>
  // 2. Negate values to the left of rightmost 1, ie:
  //    <c, b, a, 1, 0...> -> <~c, ~b, ~a, 1, 0...>
  // Next this vector can be simply reinterpreted as binary number giving
  // desired norm.
  // To handle vectorized dimensions just set all bits to one giving largest
  // representable number.
  // As a side-effect more than 31 dimensions cannot be handled with uint32_t
  // and constant dimensions always have lowest norm.
  mlir::SmallVector<uint32_t, 6> scheduleNorms;
  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    if (vector[i] != 1)
      scheduleNorms.push_back(static_cast<uint32_t>(-1));
    uint32_t norm = 0;
    mlir::AffineExpr expr = map.getResult(i);
    unsigned shMax = schedule.size();
    for (; shMax > 0; --shMax) {
      unsigned dim = schedule[shMax - 1];
      if (!expr.isFunctionOfDim(dim))
        continue;
      norm = 1;
      break;
    }
    for (unsigned sh = 0; sh < shMax; ++sh) {
      unsigned dim = schedule[sh];
      norm = (norm << 1) | !expr.isFunctionOfDim(dim);
    }
    scheduleNorms.push_back(norm);
  }

  mlir::SmallVector<unsigned, 6> dimsPermutation;
  for (unsigned i = 0; i < map.getNumResults(); ++i)
    dimsPermutation.push_back(i);

  std::stable_sort(dimsPermutation.begin(), dimsPermutation.end(),
                   [&](const unsigned &a, const unsigned &b) {
                     return scheduleNorms[a] < scheduleNorms[b];
                   });

  auto reorderMap =
      mlir::AffineMap::getPermutationMap(dimsPermutation, map.getContext());
  mlir::SmallVector<int64_t, 6> sortedShape;
  mlir::SmallVector<int64_t, 6> sortedVec;
  for (unsigned perm : dimsPermutation) {
    sortedShape.push_back(shape[perm]);
    sortedVec.push_back(vector[perm]);
  }
  return ReorderDesc{reorderMap, sortedShape, sortedVec};
}

mlir::Optional<ReorderDesc>
tileAffineMap(mlir::AffineMap map, mlir::ArrayRef<int64_t> shape,
              mlir::ArrayRef<int64_t> vector,
              mlir::FlatAffineConstraints constraints,
              mlir::ArrayRef<unsigned> schedule) {
  ReorderDesc expand = expandAffineMap(map, shape, vector, constraints);
  mlir::AffineMap expanded = expand.reorderMap.compose(map);
  mlir::AffineMap expandedSimple =
      simplifyMapWithConstraints(expanded, constraints);
  IVLOG(3, "expandedSimple: " << mlir::debugString(expandedSimple));
  mlir::ArrayRef<int64_t> expandedShape = expand.reorderedShape;
  mlir::ArrayRef<int64_t> expandedVector = expand.reorderedVector;
  ReorderDesc sort =
      sortAffineMap(expandedSimple, expandedShape, expandedVector, schedule);
  // Only sorting can change actual layout, expansion preserves indices after
  // linearization to 1D.
  if (sort.reorderMap.isIdentity())
    return llvm::None;

  return ReorderDesc{sort.reorderMap.compose(expand.reorderMap),
                     sort.reorderedShape, sort.reorderedVector};
}

std::unique_ptr<mlir::Pass> createReorderLayoutsPass() {
  return std::make_unique<ReorderLayoutsPass>();
}

std::unique_ptr<mlir::Pass>
createReorderLayoutsPass(bool allowReorder, bool makeUserLayoutsExplicit) {
  return std::make_unique<ReorderLayoutsPass>(allowReorder,
                                              makeUserLayoutsExplicit);
}

} // namespace pmlc::dialect::pxa
