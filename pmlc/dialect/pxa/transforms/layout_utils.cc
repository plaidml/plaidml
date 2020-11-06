// Copyright 2020, Intel Corporation
#include "pmlc/dialect/pxa/transforms/layout_utils.h"

#include <list>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/util/tags.h"
#include "llvm/ADT/TypeSwitch.h"

namespace pmlc::dialect::pxa {

mlir::Value createReorder(mlir::Location loc, mlir::OpBuilder &builder,
                          ReorderDesc &desc, mlir::Value srcMem) {
  // Allocate new memory.
  auto srcMemType = srcMem.getType().cast<mlir::MemRefType>();
  mlir::MemRefType newMemType =
      mlir::MemRefType::Builder(srcMemType).setShape(desc.reorderedShape);
  mlir::Value newMem = builder.create<mlir::AllocOp>(loc, newMemType);
  // Create `affine.parallel` that will perform copy with reordering.
  auto parallel = builder.create<mlir::AffineParallelOp>(
      loc, mlir::ArrayRef<mlir::Type>{newMem.getType()},
      mlir::ArrayRef<mlir::AtomicRMWKind>{mlir::AtomicRMWKind::assign},
      srcMemType.getShape());
  mlir::OpBuilder bodyBuilder = parallel.getBodyBuilder();
  mlir::MutableArrayRef<mlir::BlockArgument> loopIVs = parallel.getIVs();
  // Load with identity map.
  mlir::AffineMap identityMap =
      builder.getMultiDimIdentityMap(srcMemType.getRank());
  mlir::Value load =
      bodyBuilder.create<PxaLoadOp>(loc, srcMem, identityMap, loopIVs);
  // Assign with transformed map.
  mlir::AffineMap assignMap = desc.reorderMap.compose(identityMap);
  mlir::Value assign = bodyBuilder.create<PxaReduceOp>(
      loc, mlir::AtomicRMWKind::assign, load, newMem, assignMap, loopIVs);
  // Yield product of assignment.
  bodyBuilder.create<mlir::AffineYieldOp>(loc,
                                          mlir::ArrayRef<mlir::Value>{assign});
  return parallel.getResult(0);
}

namespace {

/// Helper class that performs use walk including indirect
/// affine uses.
/// Transforms affine maps and recreates operations to update
/// type information.
struct LayoutConverter {
  LayoutConverter(ReorderDesc &desc, mlir::MLIRContext *context)
      : reorderDesc(desc), builder(context) {}

  /// Checks whether indirect use-chain can be converted to new layout.
  bool checkCanConvert(mlir::AllocOp allocOp);
  bool checkCanConvert(mlir::Value val);

  /// Check cases for specific operation kinds.
  bool checkPxaReduceOp(PxaReduceOp op);
  bool checkPxaVectorReduceOp(PxaVectorReduceOp op);
  bool checkYieldOp(mlir::AffineYieldOp yieldOp, unsigned operandNum);

  /// Converts layout for all indirect users by transforming affine maps
  /// and recreating operations to update types.
  void convert(mlir::AllocOp allocOp);
  /// Starts use-chain walk at `val`, which is expected to already have
  /// expected data and shape.
  void convert(mlir::Value val);

  /// Convert cases for specific operation kinds.
  void convertAllocOp(mlir::AllocOp allocOp);
  void convertPxaLoadOp(PxaLoadOp loadOp);
  void convertPxaVectorLoadOp(PxaVectorLoadOp loadOp);
  void convertPxaReduceOp(PxaReduceOp reduceOp);
  void convertPxaVectorReduceOp(PxaVectorReduceOp reduceOp);
  void convertYieldOp(mlir::AffineYieldOp yieldOp, unsigned operandNum);

  /// Transforms `map` to layout specified by ReorderDesc passed to
  /// constructor.
  mlir::AffineMap transformAffineMap(mlir::AffineMap map);
  /// Transforms vector type to one specified by ReorderDesc passed
  /// to constructor.
  mlir::VectorType transformVectorType(mlir::VectorType type);

private:
  bool checkWorkQueue();
  void convertWorkQueue();

  ReorderDesc &reorderDesc;
  mlir::OpBuilder builder;
  std::list<mlir::Value> workQueue;
};

} // namespace

void replaceMemoryLayoutForReading(mlir::Value reorderedMemory,
                                   mlir::Value replaceMemory,
                                   ReorderDesc &desc) {
  // Replace uses in read operation with reordered memory.
  replaceMemory.replaceUsesWithIf(
      reorderedMemory, [&](mlir::OpOperand &operand) {
        mlir::Operation *owner = operand.getOwner();
        bool isLoad = mlir::isa<PxaLoadOp, PxaVectorLoadOp>(owner);
        bool insideReorder =
            reorderedMemory.getDefiningOp()->isAncestor(owner->getParentOp());
        return isLoad && !insideReorder;
      });
  // Update read operations using LayoutConverter.
  LayoutConverter ltConverter(desc, reorderedMemory.getContext());
  ltConverter.convert(reorderedMemory);
}

mlir::LogicalResult convertMemoryLayout(mlir::Value memory, ReorderDesc &desc) {
  auto opResult = memory.dyn_cast<mlir::OpResult>();
  if (!opResult)
    return mlir::failure();
  auto allocOp = mlir::dyn_cast<mlir::AllocOp>(opResult.getOwner());
  if (!allocOp)
    return mlir::failure();
  LayoutConverter ltConverter(desc, memory.getContext());
  // Check that indirect use-chain can be converted.
  if (!ltConverter.checkCanConvert(allocOp))
    return mlir::failure();
  // Convert layout.
  ltConverter.convert(allocOp);
  return mlir::success();
}

// =============================================================================
// LayoutConverter implementation
// =============================================================================

bool LayoutConverter::checkCanConvert(mlir::AllocOp allocOp) {
  return checkCanConvert(allocOp.getResult());
}

bool LayoutConverter::checkCanConvert(mlir::Value val) {
  workQueue.push_back(val);
  return checkWorkQueue();
}

bool LayoutConverter::checkWorkQueue() {
  while (!workQueue.empty()) {
    mlir::Value replacement = workQueue.front();
    workQueue.pop_front();

    for (mlir::OpOperand &use : replacement.getUses()) {
      mlir::Operation *user = use.getOwner();
      bool check =
          mlir::TypeSwitch<mlir::Operation *, bool>(user)
              .Case<PxaLoadOp>([&](PxaLoadOp op) { return true; })
              .Case<PxaVectorLoadOp>([&](PxaVectorLoadOp op) { return true; })
              .Case<PxaReduceOp>(
                  [&](PxaReduceOp op) { return checkPxaReduceOp(op); })
              .Case<PxaVectorReduceOp>([&](PxaVectorReduceOp op) {
                return checkPxaVectorReduceOp(op);
              })
              .Case<mlir::AffineYieldOp>([&](mlir::AffineYieldOp op) {
                return checkYieldOp(op, use.getOperandNumber());
              })
              .Default([](mlir::Operation *op) { return false; });
      if (!check)
        return false;
    }
  }
  return true;
}

bool LayoutConverter::checkPxaReduceOp(PxaReduceOp op) {
  workQueue.push_back(op.getResult());
  return true;
}

bool LayoutConverter::checkPxaVectorReduceOp(PxaVectorReduceOp op) {
  workQueue.push_back(op.getResult());
  return true;
}

bool LayoutConverter::checkYieldOp(mlir::AffineYieldOp yieldOp,
                                   unsigned operandNum) {
  mlir::Operation *parent = yieldOp.getParentOp();
  if (auto parallelOp = mlir::dyn_cast<mlir::AffineParallelOp>(parent)) {
    workQueue.push_back(parallelOp.getResult(operandNum));
    return true;
  }
  return false;
}

void LayoutConverter::convert(mlir::AllocOp allocOp) {
  convertAllocOp(allocOp);
  convertWorkQueue();
}

void LayoutConverter::convert(mlir::Value val) {
  workQueue.push_back(val);
  convertWorkQueue();
}

void LayoutConverter::convertWorkQueue() {
  while (!workQueue.empty()) {
    mlir::Value replacement = workQueue.front();
    workQueue.pop_front();

    for (mlir::OpOperand &use :
         llvm::make_early_inc_range(replacement.getUses())) {
      mlir::Operation *user = use.getOwner();
      mlir::TypeSwitch<mlir::Operation *>(user)
          .Case<PxaLoadOp>([&](PxaLoadOp op) { convertPxaLoadOp(op); })
          .Case<PxaVectorLoadOp>(
              [&](PxaVectorLoadOp op) { convertPxaVectorLoadOp(op); })
          .Case<PxaReduceOp>([&](PxaReduceOp op) { convertPxaReduceOp(op); })
          .Case<PxaVectorReduceOp>(
              [&](PxaVectorReduceOp op) { convertPxaVectorReduceOp(op); })
          .Case<mlir::AffineYieldOp>([&](mlir::AffineYieldOp op) {
            convertYieldOp(op, use.getOperandNumber());
          })
          .Default([](mlir::Operation *op) {});
    }
  }
}

void LayoutConverter::convertAllocOp(mlir::AllocOp allocOp) {
  builder.setInsertionPoint(allocOp.getOperation());

  mlir::MemRefType oldMemType = allocOp.getType();
  mlir::MemRefType newMemType = mlir::MemRefType::Builder(oldMemType)
                                    .setShape(reorderDesc.reorderedShape);

  mlir::Value newMemory =
      builder.create<mlir::AllocOp>(allocOp.getLoc(), newMemType);
  allocOp.replaceAllUsesWith(newMemory);
  allocOp.erase();
  workQueue.push_back(newMemory);
}

void LayoutConverter::convertPxaLoadOp(PxaLoadOp loadOp) {
  builder.setInsertionPoint(loadOp.getOperation());

  mlir::AffineMap newMap = transformAffineMap(loadOp.getAffineMap());
  mlir::Value loadRes = builder.create<PxaLoadOp>(
      loadOp.getLoc(), loadOp.getMemRef(), newMap, loadOp.indices());
  loadOp.replaceAllUsesWith(loadRes);
  loadOp.erase();
}

void LayoutConverter::convertPxaVectorLoadOp(PxaVectorLoadOp loadOp) {
  builder.setInsertionPoint(loadOp.getOperation());

  mlir::AffineMap newMap = transformAffineMap(loadOp.getAffineMap());
  mlir::VectorType newVectorType = transformVectorType(loadOp.getVectorType());
  mlir::Value loadRes = builder.create<PxaVectorLoadOp>(
      loadOp.getLoc(), newVectorType, loadOp.getMemRef(), newMap,
      loadOp.indices());
  loadOp.replaceAllUsesWith(loadRes);
  loadOp.erase();
}

void LayoutConverter::convertPxaReduceOp(PxaReduceOp reduceOp) {
  builder.setInsertionPoint(reduceOp.getOperation());

  mlir::AffineMap newMap = transformAffineMap(reduceOp.getAffineMap());
  mlir::Value reduceRes = builder.create<PxaReduceOp>(
      reduceOp.getLoc(), reduceOp.getAgg(), reduceOp.val(),
      reduceOp.getMemRef(), newMap, reduceOp.getMapOperands());
  reduceOp.replaceAllUsesWith(reduceRes);
  workQueue.push_back({reduceRes});
  reduceOp.erase();
}

void LayoutConverter::convertPxaVectorReduceOp(PxaVectorReduceOp reduceOp) {
  builder.setInsertionPoint(reduceOp.getOperation());

  mlir::AffineMap newMap = transformAffineMap(reduceOp.getAffineMap());
  mlir::Value reduceRes = builder.create<PxaVectorReduceOp>(
      reduceOp.getLoc(), reduceOp.getAgg(), reduceOp.vector(),
      reduceOp.getMemRef(), newMap, reduceOp.getMapOperands());
  reduceOp.replaceAllUsesWith(reduceRes);
  workQueue.push_back({reduceRes});
  reduceOp.erase();
}

void LayoutConverter::convertYieldOp(mlir::AffineYieldOp yieldOp,
                                     unsigned operandNum) {
  builder.setInsertionPoint(yieldOp.getOperation());

  auto newYield =
      builder.create<mlir::AffineYieldOp>(yieldOp.getLoc(), yieldOp.operands());

  mlir::Operation *parent = yieldOp.getParentOp();
  builder.setInsertionPoint(parent);
  if (auto parallelOp = mlir::dyn_cast<mlir::AffineParallelOp>(parent)) {
    auto newTypes = newYield.getOperands().getTypes();
    mlir::SmallVector<mlir::AtomicRMWKind, 1> reductions;
    for (mlir::Attribute attr : parallelOp.reductions()) {
      auto intAttr = attr.cast<mlir::IntegerAttr>();
      mlir::Optional<mlir::AtomicRMWKind> optReduction =
          mlir::symbolizeAtomicRMWKind(intAttr.getInt());
      reductions.push_back(optReduction.getValue());
    }
    auto newParallel = builder.create<mlir::AffineParallelOp>(
        parallelOp.getLoc(), newTypes, reductions, parallelOp.lowerBoundsMap(),
        parallelOp.getLowerBoundsOperands(), parallelOp.upperBoundsMap(),
        parallelOp.getUpperBoundsOperands(), parallelOp.getSteps());
    newParallel.region().takeBody(parallelOp.region());
    if (hasTags(parallelOp))
      copyTags(newParallel, parallelOp);
    parallelOp.replaceAllUsesWith(newParallel.getResults());
    workQueue.push_back(newParallel.getResult(operandNum));
    parent->erase();
  }
  yieldOp.erase();
}

mlir::AffineMap LayoutConverter::transformAffineMap(mlir::AffineMap map) {
  return reorderDesc.reorderMap.compose(map);
}

mlir::VectorType LayoutConverter::transformVectorType(mlir::VectorType type) {
  // TODO: Handle change in vectorization.
  return type;
}

} // namespace pmlc::dialect::pxa
