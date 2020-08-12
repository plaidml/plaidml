// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::pxa {

namespace {

struct MemRefAccess {
  AffineValueMap accessMap;

  explicit MemRefAccess(pxa::AffineLoadOp op) {
    getAccessMap(op.getAffineMap(), op.getMapOperands(), &accessMap);
  }

  explicit MemRefAccess(AffineReduceOp op) {
    getAccessMap(op.getAffineMap(), op.getMapOperands(), &accessMap);
  }

  static void getAccessMap(AffineMap map, SmallVector<Value, 8> operands,
                           AffineValueMap *accessMap) {
    fullyComposeAffineMapAndOperands(&map, &operands);
    map = simplifyAffineMap(map);
    canonicalizeMapAndOperands(&map, &operands);
    accessMap->reset(map, operands);
  }

  bool operator==(const MemRefAccess &rhs) const {
    AffineValueMap diff, lhsMap, rhsMap;
    AffineValueMap::difference(accessMap, rhs.accessMap, &diff);
    return llvm::all_of(diff.getAffineMap().getResults(),
                        [](AffineExpr expr) { return expr == 0; });
  }

  bool operator!=(const MemRefAccess &rhs) const { return !(*this == rhs); }
};

struct MemRefDataFlowOptPass
    : public MemRefDataFlowOptBase<MemRefDataFlowOptPass> {
  void runOnFunction() final {
    // Walk all load's and perform reduce to load forwarding.
    FuncOp f = getFunction();
    f.walk([&](pxa::AffineLoadOp loadOp) {
      auto defOp = loadOp.getMemRef().getDefiningOp();
      if (!defOp) {
        return;
      }

      auto reduceOp = dyn_cast_or_null<AffineReduceOp>(defOp);
      if (!reduceOp || reduceOp.agg() != AtomicRMWKind::assign) {
        return;
      }

      MemRefAccess srcAccess(reduceOp);
      MemRefAccess dstAccess(loadOp);
      IVLOG(1, "src: " << debugString(*reduceOp));
      IVLOG(1, "dst: " << debugString(*loadOp));
      if (srcAccess != dstAccess)
        return;

      // Perform the actual store to load forwarding.
      loadOp.getResult().replaceAllUsesWith(reduceOp.getValueToStore());

      loadOp.erase();
    });
  }
};

} // namespace

std::unique_ptr<Pass> createMemRefDataFlowOptPass() {
  return std::make_unique<MemRefDataFlowOptPass>();
}

} // namespace pmlc::dialect::pxa
