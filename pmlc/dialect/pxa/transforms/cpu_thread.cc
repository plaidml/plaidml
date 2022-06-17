// Copyright 2020 Intel Corporation
#include <bits/stdc++.h>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
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

  SmallVector<std::string, 4> inputShapePrefix = {"N", "IFH", "IFW", "IFC"};
  SmallVector<std::string, 5> reorderedInputShapePrefix = {"N", "IFC", "IFH",
                                                           "IFW", "IFC'"};
  SmallVector<std::string, 6> reorderedWeightShapePrefix = {"K", "IFC",  "R",
                                                            "S", "IFC'", "K'"};
  SmallVector<std::string, 4> outputShapePrefix = {"N", "OFH", "OFW", "OFC"};
  SmallVector<std::string, 5> reorderedOutputShapePrefix = {"N", "OFC", "OFH",
                                                            "OFW", "OFC'"};

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
      shape.insert(std::make_pair(reorderedWeightShapePrefix[i], typeVal));
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

  std::map<AffineParallelOp, std::map<std::pair<Block *, int>, std::string>>
  getInductionVariableLabels(AffineParallelOp op, PxaGenericOp gemmOp,
                             std::list<AffineParallelOp> parentOpList) {
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
        for (int j = 0; j < gemmOp.getNumInputs(); j++) {
          Attribute accessMap = gemmOp.inputAccessMaps()[j];
          AffineMapAttr accessMapAttr = accessMap.cast<AffineMapAttr>();
          size_t count = accessMapAttr.getValue().getNumInputs();
          auto valueRangeOp = indices.slice(prefix, count);
          prefix += count;
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
              if (gemmOp.inputAccessMaps()[0]
                      .cast<AffineMapAttr>()
                      .getValue()
                      .getNumResults() == 5) {
                inductionVarLabelsOfOp.insert(
                    std::make_pair(std::make_pair(blockArg.getOwner(),
                                                  blockArg.getArgNumber()),
                                   reorderedInputShapePrefix[index]));
              } else {
                inductionVarLabelsOfOp.insert(
                    std::make_pair(std::make_pair(blockArg.getOwner(),
                                                  blockArg.getArgNumber()),
                                   inputShapePrefix[index]));
              }
            } else {
              assert(j == 1);
              inductionVarLabelsOfOp.insert(std::make_pair(
                  std::make_pair(blockArg.getOwner(), blockArg.getArgNumber()),
                  reorderedWeightShapePrefix[index]));
            }

            break;
          }
        }
      }
      inductionVarLabels.insert(
          std::make_pair(parallelOp, inductionVarLabelsOfOp));

      for (auto opItr = parallelOp.getBody()->begin();
           opItr != parallelOp.getBody()->end(); opItr++) {
        if (isa<AffineParallelOp>(opItr)) {
          bool nestedOp = false;
          for (auto parentOp : parentOpList) {
            if (parentOp == dyn_cast<AffineParallelOp>(opItr)) {
              nestedOp = true;
              break;
            }
          }
          if (nestedOp) {
            parallelOpList.push_back(dyn_cast<AffineParallelOp>(opItr));
          }
        }
      }
    }
    return inductionVarLabels;
  }

  std::pair<AffineParallelOp, AffineParallelOp>
  splitLoop(AffineParallelOp affineParallelOp,
            std::pair<Block *, int> blockArg) {
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
            ubMap.push_back(affineParallelOp.getLowerBoundMap(i));
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
        Operation *affineYield = insideBuilder.create<AffineYieldOp>(
            newPloopWithVar.getLoc(), ValueRange{newPloop.getResult(0)});
        affineParallelOp.replaceAllUsesWith(newPloopWithVar);
        affineParallelOp.erase();
        break;
      }
    }
    return std::make_pair(newPloopWithVar, newPloop);
  }

  AffineParallelOp fuseLoops(std::list<AffineParallelOp> fusionCandidates) {
    // Find the outermost candidate first
    for (auto fusionCandidate : fusionCandidates) {
    }
    // Change stride, access functions
    return NULL;
  }

  void applyTransformations(std::list<Transformation> transformations,
                            AffineParallelOp op, PxaGenericOp gemmOp,
                            std::list<AffineParallelOp> parentOp) {
    std::map<AffineParallelOp, std::map<std::pair<Block *, int>, std::string>>
        inductionVarLabels = getInductionVariableLabels(op, gemmOp, parentOp);
    for (auto transformation : transformations) {
      switch (transformation.type) {
      case PARALLELIZE:
        for (auto inductionVar : transformation.inductionVars) {
          std::list<AffineParallelOp> fusionCandidates;
          std::list<AffineParallelOp> alreadyTraversedOps;
          for (auto affineParallelOp : inductionVarLabels) {
            if (std::find(alreadyTraversedOps.begin(),
                          alreadyTraversedOps.end(), affineParallelOp.first) ==
                alreadyTraversedOps.end()) {
              for (auto affineOpInductionVar : affineParallelOp.second) {
                if (affineOpInductionVar.second == inductionVar) {
                  AffineParallelOp fusionCandidate = affineParallelOp.first;
                  if (affineParallelOp.second.size() > 1) {
                    // Split into two such that the var in question
                    // (inductionVar) and the rest are partitioned in two
                    // separate loops
                    auto splitLoops =
                        splitLoop(fusionCandidate, affineOpInductionVar.first);
                    fusionCandidate = splitLoops.first;
                    // Find the equivalent gemm op that was cloned and the
                    // parent ops list
                    std::list<AffineParallelOp> newParentOp;
                    auto affineGemmOp =
                        getAffineOpGemm(fusionCandidate, newParentOp);
                    inductionVarLabels = getInductionVariableLabels(
                        fusionCandidate, affineGemmOp, newParentOp);
                    alreadyTraversedOps.push_back(fusionCandidate);
                    alreadyTraversedOps.push_back(splitLoops.second);
                  }
                  fusionCandidates.push_back(fusionCandidate);
                  break;
                }
              }
            }
          }
          // assert(fusionCandidates.size() > 0);
          // AffineParallelOp fusedAffineParallelOp = fusionCandidates.front();
          // if (fusionCandidates.size() > 1) {
          //    fusedAffineParallelOp = fuseLoops(fusionCandidates);
          //  }
          // Mark Affine Parallel op as parallel
          for (auto fusedAffineParallelOp : fusionCandidates) {
            setUnitTag(fusedAffineParallelOp, kCpuThreadTag);
          }
        }
        break;
      case SERIALIZE:
        break;
      case COLLAPSE:
        break;
        // Ensure that collapse variables are in the right order in the loop
        // body
      }
    }
  }

  pxa::PxaGenericOp getAffineOpGemm(AffineParallelOp op,
                                    std::list<AffineParallelOp> &parentOpList) {
    std::list<AffineParallelOp> nestedOpList;
    nestedOpList.push_back(op);
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
          return dyn_cast<pxa::PxaGenericOp>(instItr);
        }
      }
    }
    return NULL;
  }

  void runOnOperation() final {
    auto func = getOperation();
    std::list<pxa::PxaGenericOp> gemmOpsTraversed;
    // Nest outermost loops into 'blocks' and 'threads'
    func.walk<WalkOrder::PreOrder>([&](AffineParallelOp op) {
      std::list<AffineParallelOp> parentOp;
      auto affineGemmOp = getAffineOpGemm(op, parentOp);
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
          bool match = isMatchingShape(rules, affineGemmOp);
          if (match) {
            applyTransformations(transformations, op, affineGemmOp, parentOp);
          }
        }
      } else {
        processOp(op);
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

} // namespace pmlc::dialect::pxa.
