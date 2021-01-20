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

struct MemRefDataFlowOptPass
    : public MemRefDataFlowOptBase<MemRefDataFlowOptPass> {

  explicit MemRefDataFlowOptPass(bool onlyParallelNested) {
    this->onlyParallelNested = onlyParallelNested;
  }

  void runOnFunction() final {
    // Walk all load's and perform reduce to load forwarding.
    FuncOp f = getFunction();
    f.walk([&](PxaReadOpInterface loadOp) {
      auto defOp = loadOp.getMemRef().getDefiningOp();
      if (!defOp) {
        return;
      }

      auto reduceOp = dyn_cast_or_null<PxaReduceOpInterface>(defOp);
      if (!reduceOp || reduceOp.getAgg() != AtomicRMWKind::assign) {
        return;
      }

      if (onlyParallelNested &&
          (!dyn_cast<AffineParallelOp>(loadOp->getParentOp()) ||
           !dyn_cast<AffineParallelOp>(reduceOp->getParentOp()))) {
        return;
      }

      MemRefAccess srcAccess(reduceOp);
      MemRefAccess dstAccess(loadOp);
      IVLOG(3, "src: " << debugString(*reduceOp));
      IVLOG(3, "dst: " << debugString(*loadOp));
      if (srcAccess != dstAccess)
        return;

      // Perform the actual store to load forwarding.
      loadOp.getOperation()->getResult(0).replaceAllUsesWith(
          reduceOp.getValueToStore());

      loadOp.erase();
    });
  }
};

} // namespace

std::unique_ptr<Pass> createMemRefDataFlowOptPass(bool onlyParallelNested) {
  return std::make_unique<MemRefDataFlowOptPass>(onlyParallelNested);
}

} // namespace pmlc::dialect::pxa
