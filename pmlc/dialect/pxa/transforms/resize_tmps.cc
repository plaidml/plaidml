// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

namespace {

struct ResizeTmpsPass : public ResizeTmpsBase<ResizeTmpsPass> {
  explicit ResizeTmpsPass(bool onlyParallelNested) {
    this->onlyParallelNested = onlyParallelNested;
  }

  void runOnFunction() final {
    auto func = getFunction();
    func.walk([&](AllocOp op) { runOnAlloc(op); });
  }

  AffineValueMap computeInnerValueMap(AffineMap orig, ValueRange operands,
                                      Block *block) {
    auto strides = computeStrideInfo(orig, operands);
    assert(strides && "Could not compute stride info");
    SmallVector<StrideInfo, 4> inner;
    for (size_t i = 0; i < strides->size(); i++) {
      auto innerStride = (*strides)[i].inner(block);
      inner.push_back(innerStride);
    }
    return convertToValueMap(orig.getContext(), inner);
  }

  void runOnAlloc(AllocOp op) {
    Block *opBlock = op.getOperation()->getBlock();
    IVLOG(2, "Considering: " << debugString(*op.getOperation()));

    for (auto &use : getIndirectUses(op)) {
      if (isa<ReturnOp>(use.getOwner())) {
        IVLOG(2, "Found ReturnOp user, cannot resize allocation");
        return;
      } else if (isa<pmlc::dialect::stdx::ReshapeOp>(use.getOwner())) {
        IVLOG(2, "Found ReshapeOp user, cannot resize allocation");
        return;
      }
    }

    // Resize only nested ops option
    if (onlyParallelNested && !dyn_cast<AffineParallelOp>(op.getParentOp())) {
      return;
    }

    SmallVector<StrideInfo, 4> outer;
    SmallVector<StrideRange, 4> inner;
    auto vectorSize = 0;
    for (auto &use : getIndirectAccessUses(op)) {
      IVLOG(2, "Found use: " << debugString(*use.getOwner()));
      Optional<SmallVector<StrideInfo, 4>> maybeStrides;
      if (auto lop = dyn_cast<PxaReadOpInterface>(use.getOwner())) {
        maybeStrides =
            computeStrideInfo(lop.getAffineMap(), lop.getMapOperands());
        // Get vector size value. For scalar ops vec size is set to 1.
        // For now this value can be optimized only when all sizes match.
        // TODO: investigate if different values can be used here and how to
        // set the indices afterwards
        if (auto vecOp = dyn_cast<PxaVectorLoadOp>(use.getOwner())) {
          auto vecShape = vecOp.getVectorType().getShape();
          // Accept only vectors with dim size 1
          if (vecShape.size() != 1)
            return;
          // Check if size is the same as for previous op
          vectorSize =
              (vectorSize == 0 || vectorSize == vecShape[0]) ? vecShape[0] : 0;
        } else if (auto scalarOp = dyn_cast<PxaLoadOp>(use.getOwner())) {
          vectorSize = (vectorSize == 0 || vectorSize == 1) ? 1 : 0;
        }
        if (!vectorSize) {
          use.getOwner()->emitRemark(
              "Users of Alloc ops have different vector/scalar sizes");
          return;
        }
      } else if (auto rop = dyn_cast<PxaReduceOpInterface>(use.getOwner())) {
        maybeStrides =
            computeStrideInfo(rop.getAffineMap(), rop.getMapOperands());
        // Get vector size value. For scalar ops vec size is set to 1.
        // For now this value can be optimized only when all sizes match.
        // TODO: investigate if different values can be used here and how to
        // set the indices afterwards
        if (auto vecOp = dyn_cast<PxaVectorReduceOp>(use.getOwner())) {
          auto vecShape = vecOp.getVectorType().getShape();
          // Accept only vectors with dim size 1
          if (vecShape.size() != 1)
            return;
          // Check if size is the same as for previous op
          vectorSize =
              (vectorSize == 0 || vectorSize == vecShape[0]) ? vecShape[0] : 0;
        } else if (auto scalarOp = dyn_cast<PxaReduceOp>(use.getOwner())) {
          vectorSize = (vectorSize == 0 || vectorSize == 1) ? 1 : 0;
        }
        if (!vectorSize) {
          use.getOwner()->emitRemark(
              "Users of Alloc ops have different vector/scalar sizes");
          return;
        }
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

      // Expand the last size for vector ops
      if (i == oldShape.size() - 1)
        newShape[i] *= vectorSize;

      if (newShape[i] != oldShape[i]) {
        sizeChanged = true;
      }
    }
    // If it's already sized right, don't bother
    if (!sizeChanged) {
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
    // Get ops first and then replace since we modify use-def chain during
    // mutation.
    SmallVector<Operation *, 4> ops;
    for (auto &use : getIndirectAccessUses(op.getResult())) {
      ops.push_back(use.getOwner());
    }
    // Now do the actual changes.  Note, we don't bother erasing the original
    // instructions, but they get cleaned up via canonicalization
    for (Operation *op : ops) {
      if (auto rop = dyn_cast<PxaReduceOpInterface>(op)) {
        // TODO: This probably should move into some sort of utility transform,
        // but I need another example or two to generalize from
        auto vm = computeInnerValueMap(rop.getAffineMap(), rop.getMapOperands(),
                                       opBlock);
        OpBuilder replace(rop.getOperation());
        if (auto ropOp = dyn_cast<PxaReduceOp>(rop.getOperation())) {
          auto nrop = replace.create<PxaReduceOp>(
              ropOp.getLoc(), ropOp.agg(), ropOp.val(), ropOp.getMemRef(),
              vm.getAffineMap(), vm.getOperands());
          ropOp.replaceAllUsesWith(nrop.result());
        } else if (auto ropOp =
                       dyn_cast<PxaVectorReduceOp>(rop.getOperation())) {
          auto nrop = replace.create<PxaVectorReduceOp>(
              ropOp.getLoc(), ropOp.agg(), ropOp.vector(), ropOp.getMemRef(),
              vm.getAffineMap(), vm.getOperands());
          ropOp.replaceAllUsesWith(nrop.result());
        }
      }
      if (auto lop = dyn_cast<PxaReadOpInterface>(op)) {
        auto vm = computeInnerValueMap(lop.getAffineMap(), lop.getMapOperands(),
                                       opBlock);
        OpBuilder replace(lop.getOperation());
        if (auto lopOp = dyn_cast<PxaLoadOp>(lop.getOperation())) {
          auto nlop =
              replace.create<PxaLoadOp>(lopOp.getLoc(), lopOp.getMemRef(),
                                        vm.getAffineMap(), vm.getOperands());
          lopOp.replaceAllUsesWith(nlop.result());
        } else if (auto lopOp = dyn_cast<PxaVectorLoadOp>(lop.getOperation())) {
          auto nlop = replace.create<PxaVectorLoadOp>(
              lopOp.getLoc(), lopOp.getVectorType(), lopOp.getMemRef(),
              vm.getAffineMap(), vm.getOperands());
          lopOp.replaceAllUsesWith(nlop.result());
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createResizeTmpsPass(bool onlyParallelNested) {
  return std::make_unique<ResizeTmpsPass>(onlyParallelNested);
}

} // namespace pmlc::dialect::pxa
