// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Support/DebugStringHelper.h"
#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

namespace pmlc::dialect::pxa {

namespace {

struct ResizeTmpsPass : public ResizeTmpsBase<ResizeTmpsPass> {
  void runOnAlloc(mlir::AllocOp op) {
    using namespace mlir; // NOLINT
    Block *opBlock = op.getOperation()->getBlock();
    IVLOG(2, "Considering: " << debugString(*op.getOperation()));

    llvm::SmallVector<StrideInfo, 4> outer;
    llvm::SmallVector<StrideRange, 4> inner;
    for (auto &use : AccessIndirectUses(op)) {
      IVLOG(2, "Found use: " << debugString(*use.getOwner()));
      Optional<llvm::SmallVector<StrideInfo, 4>> maybeStrides;
      if (auto lop = dyn_cast<AffineLoadOp>(use.getOwner())) {
        maybeStrides =
            computeStrideInfo(lop.getAffineMap(), lop.getMapOperands());
      } else if (auto rop = dyn_cast<AffineReduceOp>(use.getOwner())) {
        maybeStrides =
            computeStrideInfo(rop.getAffineMap(), rop.getMapOperands());
      }
      if (!maybeStrides) {
        use.getOwner()->emitRemark("Unable to compute strides for access");
        return;
      }

      llvm::SmallVector<StrideInfo, 4> curOuter;
      llvm::SmallVector<StrideRange, 4> curInner;
      for (size_t i = 0; i < maybeStrides->size(); i++) {
        auto dimStride = (*maybeStrides)[i];
        auto dimStrideOuter = dimStride.outer(opBlock);
        auto dimStrideInner = dimStride.inner(opBlock);
        curOuter.push_back(dimStrideOuter);
        curInner.push_back(dimStrideInner.range());
        if (!curInner.back().valid) {
          use.getOwner()->emitRemark("Invalid inner range");
          return;
        }
      }
      // If we have set outer strides, make sure we match them
      if (outer.size()) {
        assert(curOuter.size() == outer.size() &&
               "All accesses should have the same rank");
        assert(curInner.size() == inner.size() &&
               "All accesses should have the same rank");
        if (outer != curOuter) {
          use.getOwner()->emitRemark("Mismatched out access");
          return;
        }
        for (size_t i = 0; i < inner.size(); i++) {
          inner[i].unionEquals(curInner[i]);
        }
      } else {
        // Otherwise, define new outer strides
        outer = curOuter;
        inner = curInner;
      }
    }
    assert(outer.size() == inner.size() &&
           "All accesses should have the same rank");
    // Check for lots of kinds of failures and compute new size
    bool sizeChanged = false;
    llvm::SmallVector<int64_t, 4> newShape;
    auto oldShape = op.getType().getShape();
    for (size_t i = 0; i < outer.size(); i++) {
      auto outerRange = outer[i].range();
      auto innerRange = inner[i];
      if (!outerRange.valid) {
        op.emitRemark("Invalid outer range");
        return;
      }
      if (innerRange.minVal != 0) {
        op.emitRemark("Inner range has non-trivial lower bound");
        return;
      }
      if (innerRange.stride < 0) {
        op.emitRemark("Negative strides not handled");
        return;
      }
      if (outerRange.stride && innerRange.maxVal + 1 > outerRange.stride) {
        op.emitRemark("Inner and outer ranges overlap");
        return;
      }
      newShape.push_back(innerRange.maxVal + 1);
      IVLOG(2, "Computed size:" << newShape.back());
      if (newShape[i] != oldShape[i]) {
        sizeChanged = true;
      }
    }
    // If it's already sized right, don't bother
    if (!sizeChanged) {
      op.emitRemark("Alloc is already correctly sized");
      return;
    }
    // Compute new memref type
    auto newType = MemRefType::get(newShape, op.getType().getElementType());
    op.getResult().setType(newType);
    // Update everywhere
    for (auto &use : IndirectUses(op.getResult())) {
      // A bit of overkill here
      use.get().setType(newType);
      if (auto rop = mlir::dyn_cast<AffineReduceOp>(use.getOwner())) {
        // TODO: Update affine map to remove outer strides
      }
      if (auto lop = mlir::dyn_cast<AffineLoadOp>(use.getOwner())) {
        // TODO: Update affine map to remove outer strides
      }
    }
  }

  void runOnFunction() final {
    using namespace mlir; // NOLINT
    auto func = getFunction();
    func.walk([&](AllocOp op) { runOnAlloc(op); });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createResizeTmpsPass() {
  return std::make_unique<ResizeTmpsPass>();
}

} // namespace pmlc::dialect::pxa
