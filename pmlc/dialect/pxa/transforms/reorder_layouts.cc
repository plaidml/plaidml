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
#include "llvm/ADT/TypeSwitch.h"

#include "pmlc/dialect/pxa/analysis/affine_constraints.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/ir/interfaces.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/layout_utils.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/tags.h"

namespace pmlc::dialect::pxa {

class ReorderLayoutsPass final : public ReorderLayoutsBase<ReorderLayoutsPass> {
public:
  ReorderLayoutsPass() = default;
  explicit ReorderLayoutsPass(bool allowReorder) {
    this->allowReorder = allowReorder;
  }

  void runOnFunction() {
    mlir::FuncOp func = getFunction();
    mlir::DenseMap<mlir::Value, MemoryUsageDesc> globalMemory =
        gatherGlobalMemoryDescs(func, naiveScheduleModel);
    for (auto &valueDesc : globalMemory) {
      MemoryUsageDesc &memoryDesc = valueDesc.second;
      IVLOG(3, "Optimizing layout for " << mlir::debugString(memoryDesc.value));
      mlir::Optional<ReorderDesc> optReorder =
          optimizeLayoutForReads(memoryDesc);
      if (!optReorder.hasValue()) {
        IVLOG(3, "Could not select more optimal layout");
        continue;
      }
      ReorderDesc &reorder = optReorder.getValue();
      IVLOG(3, "Optimized layout: " << mlir::debugString(reorder.reorderMap));
      if (mlir::succeeded(convertMemoryLayout(memoryDesc.value, reorder)))
        continue;
      if (!allowReorder) {
        IVLOG(3,
              "Failed to change layout in-place, separate reorder not allowed");
        continue;
      }
      IVLOG(3, "Failed to change layout in-place, inserting reorder");
      reorderMemoryReads(createReorder, reorder, memoryDesc);
    }
  }
};

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
static MemoryWriteDesc gatherWriteDesc(PxaReduceOpInterface op) {
  mlir::MemRefType memRefType = op.getMemRefType();
  mlir::ArrayRef<int64_t> shapeRef = memRefType.getShape();
  mlir::SmallVector<int64_t, 4> reduceVec(shapeRef.size(), 1);
  if (auto vecReduce = mlir::dyn_cast<PxaVectorReduceOp>(op.getOperation())) {
    auto vecType = vecReduce.getVectorType();
    mlir::ArrayRef<int64_t> vecShape = vecType.getShape();
    for (unsigned idx = 0; idx < vecShape.size(); ++idx)
      reduceVec[reduceVec.size() - vecShape.size() + idx] = vecShape[idx];
  }
  return MemoryWriteDesc{std::move(reduceVec)};
}

/// Returns MemoryUsageDesc initialized with information about `memory`,
/// without any information about its usage.
static MemoryUsageDesc getEmptyUsageDesc(mlir::Value memory) {
  auto memoryType = memory.getType().cast<mlir::MemRefType>();
  mlir::ArrayRef<int64_t> shapeRef = memoryType.getShape();
  mlir::SmallVector<int64_t, 4> shape(shapeRef.begin(), shapeRef.end());
  auto desc = MemoryUsageDesc{memory, shape};
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
    });
    parallelOp.walk([&](PxaReduceOpInterface reduce) {
      mlir::Value indirectDef = getIndirectDef(reduce.getMemRef());
      // Skip memory local to `affine.parallel`.
      if (!parallelOp.isDefinedOutsideOfLoop(indirectDef))
        return;
      MemoryUsageDesc &memoryDesc = getOrCreateGlobalDesc(indirectDef);
      memoryDesc.writes.emplace_back(gatherWriteDesc(reduce));
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

mlir::Optional<ReorderDesc>
optimizeLayoutForReads(MemoryUsageDesc &memoryDesc) {
  mlir::Optional<ReorderDesc> selectedReorder = llvm::None;
  mlir::ArrayRef<int64_t> commonVector;
  if (mlir::failed(selectCommonVectorization(memoryDesc, commonVector))) {
    IVLOG(3, "Inconsistent vectorization between reads and writes");
    return llvm::None;
  }
  for (MemoryReadDesc &readDesc : memoryDesc.reads) {
    mlir::Optional<ReorderDesc> reorder =
        tileAffineMap(readDesc.readMap, memoryDesc.shape, commonVector,
                      readDesc.dimensionConstraints, readDesc.iterationOrder);
    if (!reorder.hasValue())
      return llvm::None;
    if (!selectedReorder.hasValue()) {
      selectedReorder = reorder;
      continue;
    }
    if (selectedReorder->reorderMap != reorder->reorderMap) {
      IVLOG(3, "Inconsistent layout between reads");
      return llvm::None;
    }
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

// =============================================================================
// naiveScheduleModel - implementation.
// =============================================================================
void reorderMemoryReads(const ReorderCreator &creator, ReorderDesc &reorderDesc,
                        MemoryUsageDesc &memoryDesc) {
  mlir::DenseSet<mlir::Value> memoryToReorder;
  for (MemoryReadDesc &readDesc : memoryDesc.reads) {
    PxaReadOpInterface readOp = readDesc.readOp;
    mlir::Value readMem = readOp.getMemRef();
    memoryToReorder.insert(readMem);
  }
  for (mlir::Value originalMem : memoryToReorder) {
    mlir::OpBuilder builder(originalMem.getContext());
    builder.setInsertionPointAfterValue(originalMem);
    // TODO: It should be fused location of all reads.
    mlir::Value reorderedMem =
        creator(originalMem.getLoc(), builder, reorderDesc, originalMem);
    replaceMemoryLayoutForReading(reorderedMem, originalMem, reorderDesc);
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
  }

  expansionExprs.push_back(dimExpr);
  expandedShape.push_back(dimSize);
  expandedVec.push_back(vecSize);
}

ReorderDesc expandAffineMap(mlir::AffineMap map, mlir::ArrayRef<int64_t> shape,
                            mlir::ArrayRef<int64_t> vector,
                            mlir::FlatAffineConstraints &constraints) {
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

std::unique_ptr<mlir::Pass> createReorderLayoutsPass(bool allowReorder) {
  return std::make_unique<ReorderLayoutsPass>(allowReorder);
}

} // namespace pmlc::dialect::pxa
