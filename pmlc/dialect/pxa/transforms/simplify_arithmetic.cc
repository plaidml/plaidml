// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pxa/analysis/memref_access.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::pxa {

namespace {

/// This pass reduces redundant operations. Currently it pattern matches the
/// sequence of reduce assign with 0, followed by the reduce add,
/// which is basically side effect of the fusion pass.

// Helper function to detect if constant op is 0
bool checkIfZero(ConstantOp constantVal) {
  auto valueType = constantVal.getType();
  auto value = constantVal.value();

  // Special handling for vector type
  if (auto vectorType = valueType.dyn_cast<VectorType>()) {
    valueType = vectorType.getElementType();
    auto denseAttr = value.cast<DenseElementsAttr>();
    if (!denseAttr.isSplat())
      return false;
    value = denseAttr.getSplatValue();
  }

  // Float and integer types are supported
  if (auto floatType = valueType.dyn_cast<FloatType>()) {
    auto floatAttr = value.cast<FloatAttr>();
    if (floatAttr.getValueAsDouble() != 0.0)
      return false;
  } else if (auto intType = valueType.dyn_cast<IntegerType>()) {
    auto intAttr = value.cast<IntegerAttr>();
    if (intAttr.getInt() != 0)
      return false;
  } else {
    return false;
  }
  return true;
}

void replaceAssignLoadAdd(PxaReduceOpInterface &reduceOp) {
  // Consider only addf and addi kinds
  if (reduceOp.getAgg() != AtomicRMWKind::addf &&
      reduceOp.getAgg() != AtomicRMWKind::addi)
    return;

  auto memRefOp = reduceOp.getMemRef().getDefiningOp();
  if (!memRefOp)
    return;

  // The memref operand needs to come from reduce op assign
  auto reduceAssignOp = dyn_cast<PxaReduceOpInterface>(memRefOp);
  if (!reduceAssignOp || reduceAssignOp.getAgg() != AtomicRMWKind::assign)
    return;

  // Check if both reduce add and assign are of the same type, vector or scalar
  if ((isa<PxaReduceOp>(reduceOp.getOperation()) &&
       !isa<PxaReduceOp>(reduceAssignOp.getOperation())) ||
      (isa<PxaVectorReduceOp>(reduceOp.getOperation()) &&
       !isa<PxaVectorReduceOp>(reduceAssignOp.getOperation())))
    return;

  // Value assigned in the reduce assign op needs to come from constant op and
  // be 0
  auto assignValOp = reduceAssignOp.getValueToStore().getDefiningOp();
  if (!assignValOp)
    return;

  auto constantVal = dyn_cast<ConstantOp>(assignValOp);
  if (!constantVal || !checkIfZero(constantVal))
    return;

  // Verify if the memory access are the same (use the same method as in
  // memref dataflow opt for this)
  MemRefAccess reduceAccess(reduceOp);
  MemRefAccess reduceAssignAccess(reduceAssignOp);
  if (reduceAccess != reduceAssignAccess)
    return;

  // Make sure that reduce assign result is used only by the reduce add
  // Also check if the memref is not modified by any other operation
  if (!reduceAssignOp.getReduceResult().getUseList()->hasOneUse() ||
      !reduceAssignOp.getMemRef().getUseList()->hasOneUse())
    return;

  // Check if parent op is the same for both
  if (reduceOp->getParentOp() != reduceAssignOp->getParentOp())
    return;

  // Build new op and replace all usages
  OpBuilder builder(reduceOp);
  if (isa<PxaVectorReduceOp>(reduceOp.getOperation())) {
    auto newReduceOp = builder.create<PxaVectorReduceOp>(
        reduceOp.getLoc(), AtomicRMWKind::assign, reduceOp.getValueToStore(),
        reduceAssignOp.getMemRef(), reduceOp.getAffineMap(),
        reduceOp.getIdxs());
    reduceOp.getReduceResult().replaceAllUsesWith(newReduceOp.getResult());
  } else {
    auto newReduceOp = builder.create<PxaReduceOp>(
        reduceOp.getLoc(), AtomicRMWKind::assign, reduceOp.getValueToStore(),
        reduceAssignOp.getMemRef(), reduceOp.getAffineMap(),
        reduceOp.getIdxs());
    reduceOp.getReduceResult().replaceAllUsesWith(newReduceOp.getResult());
  }

  // Remove redundant ops
  reduceOp.erase();
  reduceAssignOp.erase();
}

struct SimplifyArithmeticPass
    : public SimplifyArithmeticBase<SimplifyArithmeticPass> {
  void runOnFunction() final {
    FuncOp f = getFunction();
    f.walk(
        [&](PxaReduceOpInterface reduceOp) { replaceAssignLoadAdd(reduceOp); });
  }
};

} // namespace

std::unique_ptr<Pass> createSimplifyArithmeticPass() {
  return std::make_unique<SimplifyArithmeticPass>();
}

} // namespace pmlc::dialect::pxa
