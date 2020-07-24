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

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

namespace {

struct ResizeTmpsPass : public ResizeTmpsBase<ResizeTmpsPass> {
  void runOnFunction() final {
    auto func = getFunction();
    func.walk([&](AllocOp op) { runOnAlloc(op); });
  }

  AffineMap computeInnerMap(AffineMap orig, ValueRange operands, Block *block) {
    auto strides = computeStrideInfo(orig, operands);
    assert(strides && "Could not compute stride info");
    SmallVector<AffineExpr, 4> newExprs;
    for (size_t i = 0; i < strides->size(); i++) {
      auto innerStrides = (*strides)[i].inner(block);
      newExprs.push_back(innerStrides.toExpr(orig.getContext(), operands));
    }
    return AffineMap::get(operands.size(), 0, newExprs, orig.getContext());
  }

  void runOnAlloc(AllocOp op) {
    Block *opBlock = op.getOperation()->getBlock();
    IVLOG(2, "Considering: " << debugString(*op.getOperation()));

    for (auto &use : getIndirectUses(op)) {
      if (isa<ReturnOp>(use.getOwner())) {
        IVLOG(2, "Found ReturnOp user, cannot resize allocation");
        return;
      }
    }

    SmallVector<StrideInfo, 4> outer;
    SmallVector<StrideRange, 4> inner;
    for (auto &use : getIndirectAccessUses(op)) {
      IVLOG(2, "Found use: " << debugString(*use.getOwner()));
      Optional<SmallVector<StrideInfo, 4>> maybeStrides;
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

      SmallVector<StrideInfo, 4> curOuter;
      SmallVector<StrideRange, 4> curInner;
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
    SmallVector<int64_t, 4> newShape;
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
      IVLOG(2, "Original size:" << oldShape[i]);
      IVLOG(2, "Computed size:" << newShape[i]);

      // if you assume that incoming IR is sane
      // then there is no need to expand
      if (newShape[i] > oldShape[i]) {
        IVLOG(2, "Expansion not allowed, resetting to original size");
        newShape[i] = oldShape[i];
      }

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
    // Update type on all definitions
    for (auto value : getIndirectValues(op.getResult())) {
      value.setType(newType);
    }
    // Update all of the access maps
    for (auto &use : getIndirectAccessUses(op.getResult())) {
      if (auto rop = dyn_cast<AffineReduceOp>(use.getOwner())) {
        auto map =
            computeInnerMap(rop.getAffineMap(), rop.getMapOperands(), opBlock);
        rop.setAttr(AffineReduceOp::getMapAttrName(), AffineMapAttr::get(map));
      }
      if (auto lop = dyn_cast<AffineLoadOp>(use.getOwner())) {
        auto map =
            computeInnerMap(lop.getAffineMap(), lop.getMapOperands(), opBlock);
        lop.setAttr(AffineLoadOp::getMapAttrName(), AffineMapAttr::get(map));
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createResizeTmpsPass() {
  return std::make_unique<ResizeTmpsPass>();
}

} // namespace pmlc::dialect::pxa
