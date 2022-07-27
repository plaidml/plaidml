// Copyright 2020 Intel Corporation
#include <bits/stdc++.h>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/RegionUtils.h"
#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/ir/matchers.h"
#include "pmlc/dialect/pxa/transforms/autotile.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/pxa/transforms/tile.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/matchers.h"
#include "pmlc/util/tags.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

namespace {
// trim from start
static inline std::string &ltrim(std::string &s) {
  s.erase(s.begin(),
          std::find_if(s.begin(), s.end(),
                       std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       std::not1(std::ptr_fun<int, int>(std::isspace)))
              .base(),
          s.end());
  return s;
}

// trim from both ends
static inline std::string &trim(std::string &s) { return ltrim(rtrim(s)); }

static constexpr llvm::StringLiteral kCpuThreadTag = "cpuThread";

// Pick the tiling that is as large as possible without going over maxThreads
struct CostModel {
  unsigned maxThreads;
  ArrayRef<int64_t> strides;

  explicit CostModel(unsigned maxThreads, ArrayRef<int64_t> strides)
      : maxThreads(maxThreads), strides(strides) {}

  double operator()(ArrayRef<int64_t> tile, double bestCost) const {
    int64_t innerSize = 1;
    int64_t maxStride = 1;
    for (size_t i = 0; i < tile.size(); i++) {
      innerSize *= tile[i];
      if (tile[i] != 1)
        maxStride = std::max(maxStride, strides[i]);
    }
    if (innerSize > maxThreads) {
      return std::numeric_limits<double>::infinity();
    }
    return (1.0 / innerSize) + (1.0 / maxStride);
  }
};

struct CPUThreadPass : public CPUThreadBase<CPUThreadPass> {
  CPUThreadPass() = default;

  explicit CPUThreadPass(unsigned threads) { this->threads = threads; }

  SmallVector<std::string, 4> inputShapePrefix = {"N", "IFH", "IFW", "IFM"};
  SmallVector<std::string, 5> reorderedInputShapePrefix = {"N", "IFM", "IFH",
                                                           "IFW", "IFM'"};
  SmallVector<std::string, 4> weightShapePrefix = {"R", "S", "IFM", "OFM"};
  SmallVector<std::string, 6> reorderedWeightShapePrefix = {
      "OFM", "IFM", "R", "S", "IFM'", "OFM'"};
  SmallVector<std::string, 4> outputShapePrefix = {"N", "OFH", "OFW", "OFM"};
  SmallVector<std::string, 5> reorderedOutputShapePrefix = {"N", "OFM", "OFH",
                                                            "OFW", "OFM'"};

  typedef enum Comparator { EQ, NEQ, LT, LTE, GT, GTE } Comparator;
  std::map<std::string, Comparator> comparatorMap = {
      {"=", EQ}, {"!=", NEQ}, {"<", LT}, {"<=", LTE}, {">", GT}, {">=", GTE}};

  typedef enum TransformationType {
    PARALLELIZE,
    COLLAPSE,
    SERIALIZE
  } TransformationType;

  std::map<std::string, TransformationType> transformationMap = {
      {"Parallelize", PARALLELIZE},
      {"Serialize", SERIALIZE},
      {"Collapse", COLLAPSE}};

  typedef struct Transformation {
    TransformationType type;
    std::list<std::string> inductionVars;
  } Transformation;

  std::map<std::string, int> getShape(pxa::PxaGenericOp gemmOp) {
    int i = 0;
    std::map<std::string, int> shape;
    bool reorderedInput =
        (gemmOp.inputs()[0].getType().cast<MemRefType>().getShape().size() == 5)
            ? true
            : false;
    for (auto typeVal :
         gemmOp.inputs()[0].getType().cast<MemRefType>().getShape()) {
      if (reorderedInput) {
        shape.insert(std::make_pair(reorderedInputShapePrefix[i], typeVal));
      } else {
        shape.insert(std::make_pair(inputShapePrefix[i], typeVal));
      }
      i++;
    }

    i = 0;
    for (auto typeVal :
         gemmOp.inputs()[1].getType().cast<MemRefType>().getShape()) {
      if (gemmOp.inputs()[1].getType().cast<MemRefType>().getShape().size() ==
          4) {
        shape.insert(std::make_pair(weightShapePrefix[i], typeVal));
      } else {
        shape.insert(std::make_pair(reorderedWeightShapePrefix[i], typeVal));
      }
      i++;
    }

    i = 0;
    for (auto typeVal :
         gemmOp.getResults().getTypes()[0].cast<MemRefType>().getShape()) {
      if (reorderedInput) {
        shape.insert(std::make_pair(reorderedOutputShapePrefix[i], typeVal));
      } else {
        shape.insert(std::make_pair(outputShapePrefix[i], typeVal));
      }
      i++;
    }
    return shape;
  }

  bool isMatchingShape(std::map<std::string, std::pair<Comparator, int>> rules,
                       pxa::PxaGenericOp gemmOp) {
    auto opShape = getShape(gemmOp);
    bool rulesMatch = true;
    for (auto rule : rules) {
      int size = opShape[rule.first];
      switch (rule.second.first) {
      case EQ:
        if (size != rule.second.second) {
          rulesMatch = false;
        }
        break;
      case NEQ:
        if (size == rule.second.second) {
          rulesMatch = false;
        }
        break;
      case LT:
        if (size >= rule.second.second) {
          rulesMatch = false;
        }
        break;
      case LTE:
        if (size > rule.second.second) {
          rulesMatch = false;
        }
        break;
      case GT:
        if (size <= rule.second.second) {
          rulesMatch = false;
        }
        break;
      case GTE:
        if (size < rule.second.second) {
          rulesMatch = false;
        }
        break;
      default:
        break;
      }

      if (!rulesMatch) {
        break;
      }
    }
    return rulesMatch;
  }

  std::map<
      AffineParallelOp,
      std::map<
          std::pair<Block *, int>,
          std::string>> inline getInductionVariableLabels(AffineParallelOp op,
                                                          PxaGenericOp gemmOp,
                                                          std::list<
                                                              AffineParallelOp>
                                                              parentOpList) {
    std::map<AffineParallelOp, std::map<std::pair<Block *, int>, std::string>>
        inductionVarLabels;
    std::list<AffineParallelOp> parallelOpList;
    parallelOpList.push_back(op);
    while (!parallelOpList.empty()) {
      AffineParallelOp parallelOp = parallelOpList.front();
      parallelOpList.pop_front();
      std::map<std::pair<Block *, int>, std::string> inductionVarLabelsOfOp;
      // Find out what labels are used at this level
      for (int i = 0; i < parallelOp.getBody()->getArguments().size(); i++) {
        auto blockArg = parallelOp.getBody()->getArguments()[i];
        auto indices = gemmOp.inputIndices();
        size_t prefix = 0;
        for (int j = 0; j < 2; j++) {
          Attribute accessMap;
          if (j == 1) {
            accessMap = gemmOp.inputAccessMaps()[1];
          } else {
            accessMap = gemmOp.outputAccessMaps()[0];
          }

          AffineMapAttr accessMapAttr = accessMap.cast<AffineMapAttr>();
          size_t count = accessMapAttr.getValue().getNumInputs();
          auto valueRangeOp = indices.slice(prefix, count);
          if (j == 0) {
            prefix += gemmOp.inputAccessMaps()[0]
                          .cast<AffineMapAttr>()
                          .getValue()
                          .getNumInputs();
          }
          AffineMap accessMapVal = accessMapAttr.getValue();
          int index = -1;
          for (int k = 0; k < accessMapVal.getNumResults(); k++) {
            std::list<AffineExpr> exprList;
            exprList.push_back(accessMapVal.getResults()[k]);
            while (!exprList.empty()) {
              auto tempExpr = exprList.front();
              exprList.pop_front();
              if (tempExpr.getKind() == AffineExprKind::DimId) {
                unsigned pos = tempExpr.cast<AffineDimExpr>().getPosition();
                if (valueRangeOp[pos] == blockArg) {
                  index = k;
                  break;
                }
              } else if (tempExpr.dyn_cast<AffineBinaryOpExpr>()) {
                exprList.push_back(
                    tempExpr.dyn_cast<AffineBinaryOpExpr>().getLHS());
                exprList.push_back(
                    tempExpr.dyn_cast<AffineBinaryOpExpr>().getRHS());
              }
            }
            if (index > -1) {
              break;
            }
          }
          if (index > -1) {
            if (j == 0) {
              if (gemmOp.outputAccessMaps()[0]
                      .cast<AffineMapAttr>()
                      .getValue()
                      .getNumResults() == 5) {
                inductionVarLabelsOfOp.insert(
                    std::make_pair(std::make_pair(blockArg.getOwner(),
                                                  blockArg.getArgNumber()),
                                   reorderedOutputShapePrefix[index]));
              } else {
                inductionVarLabelsOfOp.insert(
                    std::make_pair(std::make_pair(blockArg.getOwner(),
                                                  blockArg.getArgNumber()),
                                   outputShapePrefix[index]));
              }
            } else {
              assert(j == 1);
              if (gemmOp.inputAccessMaps()[1]
                      .cast<AffineMapAttr>()
                      .getValue()
                      .getNumResults() == 4) {
                inductionVarLabelsOfOp.insert(
                    std::make_pair(std::make_pair(blockArg.getOwner(),
                                                  blockArg.getArgNumber()),
                                   weightShapePrefix[index]));
              } else {
                inductionVarLabelsOfOp.insert(
                    std::make_pair(std::make_pair(blockArg.getOwner(),
                                                  blockArg.getArgNumber()),
                                   reorderedWeightShapePrefix[index]));
              }
            }

            break;
          }
        }
      }
      if (!inductionVarLabelsOfOp.empty()) {
        inductionVarLabels.insert(
            std::make_pair(parallelOp, inductionVarLabelsOfOp));
      }

      for (auto opItr = parallelOp.getBody()->begin();
           opItr != parallelOp.getBody()->end(); opItr++) {
        if (isa<AffineParallelOp>(opItr)) {
          bool nestedOp = std::find(parentOpList.begin(), parentOpList.end(),
                                    dyn_cast<AffineParallelOp>(opItr)) !=
                          parentOpList.end();
          if (nestedOp) {
            parallelOpList.push_back(dyn_cast<AffineParallelOp>(opItr));
          }
        }
      }
    }
    return inductionVarLabels;
  }

  std::pair<AffineParallelOp, AffineParallelOp> inline splitLoop(
      AffineParallelOp affineParallelOp, std::pair<Block *, int> blockArg) {
    AffineParallelOp newPloopWithVar, newPloop;
    for (auto iv : affineParallelOp.getIVs()) {
      if (iv.getOwner() == blockArg.first &&
          iv.getArgNumber() == blockArg.second) {
        // remove iv from affineParallelOp
        Location loc = affineParallelOp.getLoc();
        OpBuilder outsideBuilder(affineParallelOp);
        AffineMap lowerBoundMap = affineParallelOp.lowerBoundsMap();
        ValueRange lowerBoundOperands =
            affineParallelOp.getLowerBoundsOperands();
        SmallVector<AffineMap, 6> lbMap, newLbMap;
        for (int i = 0; i < lowerBoundMap.getNumResults(); i++) {
          if (i == blockArg.second) {
            newLbMap.push_back(affineParallelOp.getLowerBoundMap(i));
          } else {
            lbMap.push_back(affineParallelOp.getLowerBoundMap(i));
          }
        }

        AffineMap upperBoundMap = affineParallelOp.upperBoundsMap();
        SmallVector<AffineMap, 6> ubMap, newUbMap;
        for (int i = 0; i < upperBoundMap.getNumResults(); i++) {
          if (i == blockArg.second) {
            newUbMap.push_back(affineParallelOp.getUpperBoundMap(i));
          } else {
            ubMap.push_back(affineParallelOp.getUpperBoundMap(i));
          }
        }
        ValueRange upperBoundOperands =
            affineParallelOp.getUpperBoundsOperands();

        auto steps = affineParallelOp.getSteps();
        SmallVector<int64_t, 4> newSteps, filteredSteps;
        for (int i = 0; i < steps.size(); i++) {
          if (i == blockArg.second) {
            newSteps.push_back(steps[i]);
          } else {
            filteredSteps.push_back(steps[i]);
          }
        }
        auto tensorType =
            affineParallelOp.getResult(0).getType().cast<MemRefType>();
        auto memRefType =
            MemRefType::get(tensorType.getShape(), tensorType.getElementType());

        SmallVector<Type, 2> reductionTypes;
        reductionTypes.push_back(memRefType);
        SmallVector<arith::AtomicRMWKind, 3> reductionAttr;
        for (Attribute attr : affineParallelOp.reductions()) {
          auto intAttr = attr.dyn_cast<IntegerAttr>();
          arith::AtomicRMWKind sym =
              *arith::symbolizeAtomicRMWKind(intAttr.getInt());
          reductionAttr.push_back(sym);
        }
        newPloopWithVar = outsideBuilder.create<AffineParallelOp>(
            loc, reductionTypes, reductionAttr, llvm::makeArrayRef(newLbMap),
            lowerBoundOperands, llvm::makeArrayRef(newUbMap),
            upperBoundOperands, llvm::makeArrayRef(newSteps));

        OpBuilder insideBuilder(newPloopWithVar.getBody()->getParent());
        newPloop = insideBuilder.create<AffineParallelOp>(
            newPloopWithVar.getLoc(), reductionTypes, reductionAttr,
            llvm::makeArrayRef(lbMap), lowerBoundOperands,
            llvm::makeArrayRef(ubMap), upperBoundOperands,
            llvm::makeArrayRef(filteredSteps));
        // clone instructions into newPloop and patch arguments
        std::map<Operation *, Operation *> cloneMap;
        std::list<Operation *> clonedInstList;
        OpBuilder innermostBuilder(newPloop.getBody()->getParent());

        for (auto inst = affineParallelOp.getBody()->begin();
             inst != affineParallelOp.getBody()->end(); inst++) {
          auto clonedInst = innermostBuilder.clone(*inst);
          if (dyn_cast<AffineParallelOp>(clonedInst)) {
            std::list<AffineParallelOp> clonedAffineOps;
            clonedAffineOps.push_back(dyn_cast<AffineParallelOp>(clonedInst));
            while (!clonedAffineOps.empty()) {
              auto clonedAffineOp = clonedAffineOps.front();
              clonedAffineOps.pop_front();
              for (auto clonedAffineInst = clonedAffineOp.getBody()->begin();
                   clonedAffineInst != clonedAffineOp.getBody()->end();
                   clonedAffineInst++) {
                if (dyn_cast<AffineParallelOp>(clonedAffineInst)) {
                  clonedAffineOps.push_back(
                      dyn_cast<AffineParallelOp>(clonedAffineInst));
                }
                clonedInstList.push_back(&*clonedAffineInst);
              }
            }
          } else {
            clonedInstList.push_back(clonedInst);
          }
          cloneMap[&*inst] = clonedInst;
        }
        for (auto clonedInstrItr : clonedInstList) {
          for (auto instrItr : cloneMap) {
            for (auto index = 0; index < instrItr.first->getNumResults();
                 index++) {
              clonedInstrItr->replaceUsesOfWith(
                  instrItr.first->getResult(index),
                  cloneMap[instrItr.first]->getResult(index));
            }
          }
        }
        for (auto instrItr : clonedInstList) {
          for (int i = 0; i < instrItr->getNumOperands(); i++) {
            Value operand = instrItr->getOperand(i);
            if (std::find(affineParallelOp.getIVs().begin(),
                          affineParallelOp.getIVs().end(),
                          operand) != affineParallelOp.getIVs().end()) {
              int argIndex =
                  std::find(affineParallelOp.getIVs().begin(),
                            affineParallelOp.getIVs().end(), operand) -
                  affineParallelOp.getIVs().begin();
              if (argIndex == blockArg.second) {
                instrItr->replaceUsesOfWith(operand,
                                            newPloopWithVar.getIVs().front());
              } else {
                if (argIndex < blockArg.second) {
                  instrItr->replaceUsesOfWith(operand,
                                              newPloop.getIVs()[argIndex]);
                } else {
                  instrItr->replaceUsesOfWith(operand,
                                              newPloop.getIVs()[argIndex - 1]);
                }
              }
            }
          }
        }
        insideBuilder.create<AffineYieldOp>(newPloopWithVar.getLoc(),
                                            ValueRange{newPloop.getResult(0)});
        affineParallelOp.replaceAllUsesWith(newPloopWithVar);
        affineParallelOp.erase();
        break;
      }
    }
    return std::make_pair(newPloopWithVar, newPloop);
  }

  AffineParallelOp fuseLoops(std::list<AffineParallelOp> loops) {
    AffineParallelOp innermost = loops.back();
    AffineParallelOp outermost = loops.front();
    AffineMap origUbMap = outermost.upperBoundsMap();
    Location loc = outermost.getLoc();
    OpBuilder builder(outermost);

    SmallVector<Value, 4> upperBoundSymbols;
    SmallVector<Value, 4> ubOperands(outermost.getUpperBoundsOperands().begin(),
                                     outermost.getUpperBoundsOperands().end());
    Value prev;
    if (!llvm::hasSingleElement(origUbMap.getResults()))
      prev = builder.create<AffineMinOp>(loc, origUbMap, ubOperands);
    else
      prev = builder.create<AffineApplyOp>(loc, origUbMap, ubOperands);
    upperBoundSymbols.push_back(prev);

    loops.pop_front();
    for (AffineParallelOp loop : loops) {
      origUbMap = loop.upperBoundsMap();
      ubOperands = loop.getUpperBoundsOperands();
      Value upperBound;
      if (!llvm::hasSingleElement(origUbMap.getResults()))
        upperBound = builder.create<AffineMinOp>(loc, origUbMap, ubOperands);
      else
        upperBound = builder.create<AffineApplyOp>(loc, origUbMap, ubOperands);
      upperBoundSymbols.push_back(upperBound);
      SmallVector<Value, 4> operands;
      operands.push_back(prev);
      operands.push_back(upperBound);
      prev = builder.create<AffineApplyOp>(
          loc,
          AffineMap::get(1, 1,
                         builder.getAffineDimExpr(0) *
                             builder.getAffineSymbolExpr(0)),
          operands);
    }
    AffineMap newUbMap = AffineMap::get(0, 1, builder.getAffineSymbolExpr(0),
                                        builder.getContext());
    outermost.setUpperBounds(prev, newUbMap);
    builder.setInsertionPointToStart(outermost.getBody());
    Value previous = outermost.getIVs()[0];

    auto itr = std::next(loops.rbegin(), 1);
    for (unsigned idx = loops.size(); idx > 0; --idx) {
      if (idx != loops.size()) {
        SmallVector<Value, 4> operands;
        operands.push_back(previous);
        operands.push_back(upperBoundSymbols[idx]);
        previous = builder.create<AffineApplyOp>(
            loc,
            AffineMap::get(1, 1,
                           builder.getAffineDimExpr(0).floorDiv(
                               builder.getAffineSymbolExpr(0))),
            operands);
      }
      Value inductionVariable;
      if (idx == 1) {
        inductionVariable = previous;
      } else {
        SmallVector<Value, 4> applyOperands;
        applyOperands.push_back(previous);
        applyOperands.push_back(upperBoundSymbols[idx - 1]);
        inductionVariable = builder.create<AffineApplyOp>(
            loc,
            AffineMap::get(1, 1,
                           builder.getAffineDimExpr(0) %
                               builder.getAffineSymbolExpr(0)),
            applyOperands);
      }
      auto bb = std::prev(itr, 1);
      replaceAllUsesInRegionWith((&*bb)->getIVs()[0], inductionVariable,
                                 loops.back().region());
      itr = std::next(itr, 1);
    }
    for (auto loopItr = loops.begin(); loopItr != loops.end(); loopItr++) {
      (&*loopItr)->setUpperBounds((&*loopItr)->getLowerBoundsOperands(),
                                  (&*loopItr)->lowerBoundsMap());
    }
    return outermost;
  }

  void interchangeLoops(AffineParallelOp forOpA, AffineParallelOp forOpB) {
    assert(&*forOpA.getBody()->begin() == forOpB.getOperation());
    auto &forOpABody = forOpA.getBody()->getOperations();
    auto &forOpBBody = forOpB.getBody()->getOperations();

    // 1) Splice forOpA's non-terminator operations (which is just forOpB) just
    // before forOpA (in ForOpA's parent's block) this should leave 'forOpA's
    // body containing only the terminator.
    forOpA->getBlock()->getOperations().splice(Block::iterator(forOpA),
                                               forOpABody, forOpABody.begin(),
                                               std::prev(forOpABody.end()));
    // 2) Splice forOpB's non-terminator operations into the beginning of
    // forOpA's body (this leaves forOpB's body containing only the terminator).
    forOpABody.splice(forOpABody.begin(), forOpBBody, forOpBBody.begin(),
                      std::prev(forOpBBody.end()));
    // 3) Splice forOpA into the beginning of forOpB's body.
    forOpBBody.splice(forOpBBody.begin(), forOpA->getBlock()->getOperations(),
                      Block::iterator(forOpA));
  }

  bool isPerfectLoopNest(AffineParallelOp outerLoop,
                         AffineParallelOp innerLoop) {
    AffineParallelOp parent = innerLoop;
    AffineParallelOp prev = NULL;
    while (parent != NULL) {
      prev = parent;
      parent = dyn_cast<AffineParallelOp>(
          parent.getBody()->getParentOp()->getParentOp());
      if (!parent.getBody()->empty() &&
          prev.getOperation() != (&*parent.getBody()->begin())) {
        return false;
      }
      for (auto itr = parent.getBody()->begin(); itr != parent.getBody()->end();
           itr++) {
        if (isa<AffineParallelOp>(itr) &&
            dyn_cast<AffineParallelOp>(itr) != prev) {
          return false;
        }
      }
      if (parent == outerLoop) {
        return true;
      }
    }
    return false;
  }

  bool outerLoopOf(AffineParallelOp firstLoop, AffineParallelOp secondLoop) {
    AffineParallelOp parent = secondLoop;
    while (parent != NULL) {
      parent = dyn_cast<AffineParallelOp>(
          parent.getBody()->getParentOp()->getParentOp());
      if (parent == firstLoop) {
        return true;
      }
    }
    return false;
  }

  AffineParallelOp
  applyTransformations(std::list<Transformation> transformations,
                       AffineParallelOp op, PxaGenericOp gemmOp,
                       std::list<AffineParallelOp> parentOp) {
    std::map<AffineParallelOp, std::map<std::pair<Block *, int>, std::string>>
        inductionVarLabels = getInductionVariableLabels(op, gemmOp, parentOp);
    auto tempTopOp = op;
    for (auto transformation : transformations) {
      std::map<std::string, AffineParallelOp> inductionVarLoopMap;
      for (auto inductionVar : transformation.inductionVars) {
        std::list<AffineParallelOp> fusionCandidates;
        std::list<AffineParallelOp> alreadyTraversedOps;
        auto affineParallelOp = inductionVarLabels.begin();

        for (; affineParallelOp != inductionVarLabels.end();) {
          bool reset = false;
          if (std::find(alreadyTraversedOps.begin(), alreadyTraversedOps.end(),
                        affineParallelOp->first) == alreadyTraversedOps.end()) {
            for (auto affineOpInductionVar : affineParallelOp->second) {
              if (affineOpInductionVar.second == inductionVar) {
                AffineParallelOp fusionCandidate = affineParallelOp->first;
                if (affineParallelOp->second.size() > 1) {
                  // Split into two such that the var in question
                  // (inductionVar) and the rest are partitioned in two
                  // separate loops
                  auto splitLoops =
                      splitLoop(fusionCandidate, affineOpInductionVar.first);
                  if (fusionCandidate == tempTopOp) {
                    tempTopOp = splitLoops.first;
                  }
                  fusionCandidate = splitLoops.first;
                  // Find the equivalent gemm op that was cloned and the
                  // parent ops list
                  auto affineGemmOp = getAffineOpGemm(tempTopOp, parentOp);
                  inductionVarLabels = getInductionVariableLabels(
                      tempTopOp, affineGemmOp, parentOp);
                  affineParallelOp = inductionVarLabels.begin();
                  reset = true;
                  alreadyTraversedOps.push_back(splitLoops.first);
                  alreadyTraversedOps.push_back(splitLoops.second);
                }
                fusionCandidates.push_back(fusionCandidate);
                break;
              }
            }
          }
          if (!reset)
            affineParallelOp++;
        }
        if (fusionCandidates.size() > 0) {
          AffineParallelOp fusedAffineParallelOp = fusionCandidates.front();
          if (fusionCandidates.size() > 1) {
            fusedAffineParallelOp = fuseLoops(fusionCandidates);
          }

          if (transformation.type == PARALLELIZE) {
            // Mark Affine Parallel op as parallel
            setUnitTag(fusedAffineParallelOp, kCpuThreadTag);
          } else if (transformation.type == SERIALIZE) {
            clearTag(fusedAffineParallelOp, kCpuThreadTag);
          }
          inductionVarLoopMap.insert(
              std::make_pair(inductionVar, fusedAffineParallelOp));
        }
      }
      if (transformation.type == COLLAPSE) {
        assert(inductionVarLoopMap.size() <= 2);
        std::string firstIV = transformation.inductionVars.front();
        AffineParallelOp firstLoop, secondLoop;
        if (inductionVarLoopMap.find(firstIV) != inductionVarLoopMap.end()) {
          firstLoop = inductionVarLoopMap[firstIV];
        }
        std::string secondIV =
            *std::next(transformation.inductionVars.begin(), 1);
        if (inductionVarLoopMap.find(secondIV) != inductionVarLoopMap.end()) {
          secondLoop = inductionVarLoopMap[secondIV];
        }
        if (firstLoop != NULL && secondLoop != NULL) {
          bool isPerfectLoop = true;
          bool firstLoopOuter = outerLoopOf(firstLoop, secondLoop);
          if (firstLoopOuter) {
            isPerfectLoop = isPerfectLoopNest(firstLoop, secondLoop);
          } else {
            isPerfectLoop = isPerfectLoopNest(secondLoop, firstLoop);
          }
          // swap the second loop with ancestors till the time first and
          // second loops are placed right next to each other
          if (isPerfectLoop) {
            while (firstLoop !=
                   dyn_cast<AffineParallelOp>(
                       secondLoop.getBody()->getParentOp()->getParentOp())) {
              if (firstLoopOuter) {
                interchangeLoops(
                    dyn_cast<AffineParallelOp>(
                        secondLoop.getBody()->getParentOp()->getParentOp()),
                    secondLoop);
              } else {
                interchangeLoops(
                    dyn_cast<AffineParallelOp>(
                        firstLoop.getBody()->getParentOp()->getParentOp()),
                    firstLoop);
              }
            }
            setUnitTag(firstLoop, kCpuThreadTag);
            setIntegerTag(firstLoop, "collapse", 2);
          }
        } else if (firstLoop != NULL) {
          setUnitTag(firstLoop, kCpuThreadTag);
        } else if (secondLoop != NULL) {
          setUnitTag(secondLoop, kCpuThreadTag);
        }
      }
    }
    return tempTopOp;
  }

  pxa::PxaGenericOp getAffineOpGemm(AffineParallelOp op,
                                    std::list<AffineParallelOp> &parentOpList) {
    std::list<AffineParallelOp> nestedOpList;
    nestedOpList.push_back(op);
    parentOpList.clear();
    pxa::PxaGenericOp gemmOp = NULL;
    while (!nestedOpList.empty()) {
      AffineParallelOp nestedOp = nestedOpList.front();
      nestedOpList.pop_front();
      parentOpList.push_back(nestedOp);
      for (auto instItr = nestedOp.getBody()->begin();
           instItr != nestedOp.getBody()->end(); instItr++) {
        if (isa<AffineParallelOp>(instItr)) {
          nestedOpList.push_back(dyn_cast<AffineParallelOp>(instItr));
        } else if (isa<pxa::PxaGenericOp>(instItr) &&
                   dyn_cast<PxaGenericOp>(instItr).kernel().str() ==
                       "tpp_gemm") {
          gemmOp = dyn_cast<pxa::PxaGenericOp>(instItr);
        }
      }
    }
    return gemmOp;
  }

  void runOnOperation() final {
    auto func = getOperation();
    std::list<pxa::PxaGenericOp> gemmOpsTraversed;
    // Nest outermost loops into 'blocks' and 'threads'
    func.walk<WalkOrder::PreOrder>([&](AffineParallelOp op) {
      std::list<AffineParallelOp> parentOp;
      auto affineGemmOp = getAffineOpGemm(op, parentOp);
      bool match = false;
      AffineParallelOp topAffineOp = op;
      // Parse the file containing parallelization rules
      if (!util::getEnvVar("PLAIDML_PARALLELIZATION_CONFIG_FILE").empty() &&
          affineGemmOp != NULL &&
          std::find(gemmOpsTraversed.begin(), gemmOpsTraversed.end(),
                    affineGemmOp) == gemmOpsTraversed.end()) {
        gemmOpsTraversed.push_back(affineGemmOp);
        auto configFileName =
            util::getEnvVar("PLAIDML_PARALLELIZATION_CONFIG_FILE");
        IVLOG(1, "Configuration file name:" << configFileName);
        std::ifstream configFile(configFileName, std::ifstream::binary);
        std::string fileConfigString;
        int lineno = 0;

        while (std::getline(configFile, fileConfigString)) {
          std::stringstream configStringStream;
          configStringStream << fileConfigString;
          std::string configString;

          std::map<std::string, std::pair<Comparator, int>> rules;
          std::list<Transformation> transformations;
          int bracketCount = 0;
          while (getline(configStringStream, configString, ';')) {
            if (configString.find("[") != std::string::npos) {
              bracketCount++;
            }
            std::replace(configString.begin(), configString.end(), '[', ' ');
            std::replace(configString.begin(), configString.end(), ']', ' ');
            configString = trim(configString);
            if (bracketCount == 2) {
              std::stringstream transformStringStream;
              transformStringStream << configString;
              std::string transformString;
              int iter = 0;
              TransformationType transformerType;
              std::list<std::string> iterVars;
              while (getline(transformStringStream, transformString, ' ')) {
                if (iter == 0) {
                  transformerType = transformationMap[transformString];
                } else {
                  std::stringstream ivStr;
                  ivStr << transformString;
                  std::string iv;
                  while (getline(ivStr, iv, ',')) {
                    iterVars.push_back(iv);
                  }
                }
                iter++;
              }
              if (iter > 0) {
                Transformation tx;
                tx.type = transformerType;
                tx.inductionVars = iterVars;
                transformations.push_back(tx);
              }
            } else {
              const char *charConfigStr = configString.c_str();
              std::string lhs = "", rhs = "", symbol = "";
              for (int i = 0; i < configString.size(); i++) {
                if (charConfigStr[i] - 'A' >= 0 &&
                    'Z' - charConfigStr[i] >= 0) {
                  lhs.push_back(charConfigStr[i]);
                } else if (charConfigStr[i] - '0' >= 0 &&
                           '9' - charConfigStr[i] >= 0) {
                  rhs.push_back(charConfigStr[i]);
                } else {
                  symbol.push_back(charConfigStr[i]);
                }
              }
              Comparator comparatorVal = comparatorMap[symbol];
              rules.insert(std::make_pair(
                  lhs, std::make_pair(comparatorVal, atoi(rhs.c_str()))));
            }
          }
          auto gemmOp = getAffineOpGemm(topAffineOp, parentOp);
          match = isMatchingShape(rules, gemmOp);
          if (match) {
            topAffineOp = applyTransformations(transformations, topAffineOp,
                                               gemmOp, parentOp);
            break;
          }
        }
      }
      if (affineGemmOp == NULL) {
        processOp(topAffineOp);
      }
      return WalkResult::skip();
    });
  }

  void processOp(AffineParallelOp op) {
    auto maybeRanges = op.getConstantRanges();
    if (!maybeRanges) {
      // Fail if we can't compute the ranges at compile time
      return;
    }

    SmallVector<int64_t> strides(op.getNumDims(), 0);
    if (auto lastWriter =
            dyn_cast_or_null<PxaReduceOp>(getPrevWriter(op.getResult(0)))) {
      if (Optional<StrideInfo> si = computeStrideInfo(lastWriter)) {
        for (BlockArgument arg : op.getIVs()) {
          strides[arg.getArgNumber()] = si->strides[arg];
        }
      }
    }

    CostModel model(threads, strides);
    auto tileSize =
        findBestTileSize(EvenTilingGenerator(), model, *maybeRanges);
    // Invert tiling (we want 'threads' on the outer loop
    for (size_t i = 0; i < tileSize.size(); i++) {
      tileSize[i] = (*maybeRanges)[i] / tileSize[i];
    }
    // Tile and tag
    performTiling(op, tileSize);
    setUnitTag(op, kCpuThreadTag);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createCPUThreadPass() {
  return std::make_unique<CPUThreadPass>();
}

std::unique_ptr<mlir::Pass> createCPUThreadPass(unsigned threads) {
  return std::make_unique<CPUThreadPass>(threads);
}

} // namespace pmlc::dialect::pxa
