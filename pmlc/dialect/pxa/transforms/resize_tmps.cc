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
  void runOnFunction() final {
    using namespace mlir; // NOLINT
    auto func = getFunction();
    func.walk([&](AllocOp op) {
      Block *opBlock = op.getOperation()->getBlock();
      IVLOG(1, "Considering: " << debugString(*op.getOperation()));
      bool valid = true;
      llvm::SmallVector<StrideInfo, 4> outer;
      llvm::SmallVector<StrideRange, 4> inner;
      for (auto &use : AccessIndirectUses(op)) {
        IVLOG(1, "Found use: " << debugString(*use.getOwner()));
        Optional<llvm::SmallVector<StrideInfo, 4>> maybeStrides;
        if (auto lop = dyn_cast<AffineLoadOp>(use.getOwner())) {
          maybeStrides =
              computeStrideInfo(lop.getAffineMap(), lop.getMapOperands());
        } else if (auto rop = dyn_cast<AffineReduceOp>(use.getOwner())) {
          maybeStrides =
              computeStrideInfo(rop.getAffineMap(), rop.getMapOperands());
        }
        if (!maybeStrides) {
          IVLOG(1, "Failed since one access cannot compute strides");
          valid = false;
          break;
        }
        llvm::SmallVector<StrideInfo, 4> curOuter;
        llvm::SmallVector<StrideRange, 4> curInner;
        for (size_t i = 0; i < maybeStrides->size(); i++) {
          auto dimStride = (*maybeStrides)[i];
          auto dimStrideOuter = dimStride.outer(opBlock);
          auto dimStrideInner = dimStride.inner(opBlock);
          curOuter.push_back(dimStrideOuter);
          curInner.push_back(dimStrideInner.range());
        }
        // If we have set outer strides, make sure we match them
        if (outer.size()) {
          assert(curOuter.size() == outer.size() &&
                 "All accesses should have the same rank");
          assert(curInner.size() == inner.size() &&
                 "All accesses should have the same rank");
          if (outer != curOuter) {
            IVLOG(1, "Different outer access, cannot resize");
            valid = false;
            break;
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
      IVLOG(1, "valid = " << valid);
      for (size_t i = 0; i < inner.size(); i++) {
        IVLOG(1, "Inner " << i << ": min = " << inner[i].minVal << ", max = "
                          << inner[i].maxVal << ", stride = " << inner[i].stride
                          << ", valid = " << inner[i].valid);
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createResizeTmpsPass() {
  return std::make_unique<ResizeTmpsPass>();
}

} // namespace pmlc::dialect::pxa
