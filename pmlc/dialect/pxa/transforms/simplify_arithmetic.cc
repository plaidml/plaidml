// Copyright 2020 Intel Corporation

#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

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
bool checkIfZero(arith::ConstantOp constantVal) {
  auto valueType = constantVal.getType();
  auto value = constantVal.getValue();

  // Special handling for vector type
  if (auto vectorType = valueType.dyn_cast<VectorType>()) {
    valueType = vectorType.getElementType();
    auto denseAttr = value.cast<DenseElementsAttr>();
    if (!denseAttr.isSplat())
      return false;
    if ((denseAttr.getType().getElementType().isa<IntegerType>()) && (denseAttr.getSplatValue<APInt>().isZero()))
      return true;
    if ((denseAttr.getType().getElementType().isa<FloatType>()) && (denseAttr.getSplatValue<APFloat>().isZero()))
      return true;
  }

  // Float and integer types are supported
  if (auto floatType = valueType.dyn_cast<FloatType>()) {
    auto floatAttr = value.cast<FloatAttr>();
    if (floatAttr.getValueAsDouble() == 0.0)
      return true;
  } 
  if (auto intType = valueType.dyn_cast<IntegerType>()) {
    auto intAttr = value.cast<IntegerAttr>();
    if (intAttr.getInt() == 0)
      return true;
  }
  return false;
}

void replaceAssignLoadAdd(PxaReduceOpInterface &reduceOp) {
  // Consider only addf and addi kinds
  if (reduceOp.getAgg() != arith::AtomicRMWKind::addf &&
      reduceOp.getAgg() != arith::AtomicRMWKind::addi)
    return;

  auto memRefOp = reduceOp.getMemRef().getDefiningOp();
  if (!memRefOp)
    return;

  // The memref operand needs to come from reduce op assign
  auto reduceAssignOp = dyn_cast<PxaReduceOpInterface>(memRefOp);
  if (!reduceAssignOp || reduceAssignOp.getAgg() != arith::AtomicRMWKind::assign)
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

  auto constantVal = dyn_cast<arith::ConstantOp>(assignValOp);
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
  if (!reduceAssignOp.getReduceResult().hasOneUse() ||
      !reduceAssignOp.getMemRef().hasOneUse())
    return;

  // Check if parent op is the same for both
  if (reduceOp->getParentOp() != reduceAssignOp->getParentOp())
    return;

  // Build new op and replace all usages
  OpBuilder builder(reduceOp);
  if (isa<PxaVectorReduceOp>(reduceOp.getOperation())) {
    auto newReduceOp = builder.create<PxaVectorReduceOp>(
        reduceOp.getLoc(), arith::AtomicRMWKind::assign, reduceOp.getValueToStore(),
        reduceAssignOp.getMemRef(), reduceOp.getAffineMap(),
        reduceOp.getIdxs());
    reduceOp.getReduceResult().replaceAllUsesWith(newReduceOp.getResult());
  } else {
    auto newReduceOp = builder.create<PxaReduceOp>(
        reduceOp.getLoc(), arith::AtomicRMWKind::assign, reduceOp.getValueToStore(),
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
  void runOnOperation() final {
    func::FuncOp f = getOperation();
    f.walk(
        [&](PxaReduceOpInterface reduceOp) { replaceAssignLoadAdd(reduceOp); });
  }
};

} // namespace

std::unique_ptr<Pass> createSimplifyArithmeticPass() {
  return std::make_unique<SimplifyArithmeticPass>();
}

} // namespace pmlc::dialect::pxa
