// Copyright 2020 Intel Corporation

#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Pass/Pass.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "pmlc/dialect/pxa/transforms/tile.h"

#include "mlir/Support/DebugStringHelper.h"
#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

namespace {
using llvm::DenseMap;
using llvm::DenseSet;
using llvm::SmallVector;
using mlir::BlockArgument;

unsigned getNestedIVCount(AffineParallelOp op) {
  unsigned count = 0;
  while (op) {
    count += op.getIVs().size();
    op = dyn_cast<AffineParallelOp>(op.getBody()->front());
  }
  return count;
}

void buildNestLoop(AffineParallelOp op) {
  OpBuilder builder(op.getBody(), op.getBody()->begin());
  Block *outerBody = op.getBody();
  llvm::SmallVector<mlir::AtomicRMWKind, 8> reductions;
  for (Attribute attr : op.reductions()) {
    auto intAttr = attr.dyn_cast<IntegerAttr>();
    reductions.push_back(*mlir::symbolizeAtomicRMWKind(intAttr.getInt()));
  }
  auto inner = builder.create<AffineParallelOp>(
      op.getLoc(), op.getResultTypes(), reductions, ArrayRef<int64_t>{1});
  // Splice instructions into the interior
  auto &innerLoopOps = inner.getBody()->getOperations();
  auto &outerLoopOps = outerBody->getOperations();
  innerLoopOps.splice(std::prev(innerLoopOps.end()), outerLoopOps,
                      std::next(outerLoopOps.begin(), 1), outerLoopOps.end());
  // Add a return of the values of the inner to the outer
  builder.setInsertionPointToEnd(op.getBody());
  builder.create<AffineYieldOp>(op.getLoc(), inner.getResults());
}

struct NestLoopsPass : public NestLoopsBase<NestLoopsPass> {
  NestLoopsPass() = default;
  explicit NestLoopsPass(unsigned minLoopIVs) : minLoopIVs(minLoopIVs) {}
  void runOnFunction() final {
    auto func = getFunction();
    // Nest output loops
    for (auto op : func.getBody().getOps<AffineParallelOp>()) {
      while (getNestedIVCount(op) < minDepth) {
        buildNestLoop(op);
      }
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createNestLoopPass() {
  return std::make_unique<NestLoopsPass>();
}

std::unique_ptr<mlir::Pass> createNestLoopPass(unsigned minLoopIVs) {
  return std::make_unique<NestLoopsPass>(minLoopIVs);
}

} // namespace pmlc::dialect::pxa.
