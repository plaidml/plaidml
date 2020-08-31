// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/stdx/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::stdx {

namespace {

/// Changes load and store operations withing subgroups into
/// stdx.subgroup_block_read_intel and stdx.subgroup_block_write_intel ops,
/// where possible. This pass should be performed after subgroup broadcast
/// pass which also performs devectorization.
///
/// Sample code:
///
///  %c0 = constant 0 : index
///  %c8 = constant 8 : index
///  %c64 = constant 64 : index
///  %cst = constant 0.000000e+00 : f32
///  %c1 = constant 1 : index
///  scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c64, %c8) step (%c8, %c1) {
///    %0 = addi %arg2, %arg3 : index
///    %1 = load %arg0[%0] : memref<64xf32>
///    store %1, %arg1[%0] : memref<64xf32>
///    scf.yield
///  }
///
/// Code after transformation:
///
///  %c0 = constant 0 : index
///  %c1 = constant 1 : index
///  %c8 = constant 8 : index
///  %c64 = constant 64 : index
///  scf.parallel (%i, %sid) = (%c0, %c0) to (%c64, %c8) step (%c8, %c1) {
///    %idx = addi %i, %sid : index
///    %0 = stdx.subgroup_block_read_intel %arg0[%i] : memref<64xf32>
///    stdx.subgroup_block_write_intel %0, %arg1[%i] : memref<64xf32>
///    scf.yield

class BlockOpsImpl {
private:
  scf::ParallelOp loop;
  DenseSet<Operation *> toReplace;

  template <typename OpType>
  void checkLoadStore(OpType loadOp) {
    // The loadOp/storeOp needs to be stride 1,
    // and the last index needs to be from AddI operation with subgroup_id
    // (this is modelled in subgroup_broadcast pass)
    //
    // TODO: Verify if this condition is correct, or it
    // will be needed to check for the additional parallel
    // op gpu thread attribute
    if (auto addIOp =
            dyn_cast<AddIOp>(loadOp.indices().back().getDefiningOp())) {
      auto addIOperand = addIOp.getOperand(1);
      auto lastIndiceOfLoop = loop.getBody()->getArguments().back();
      int64_t memRefOffset;
      SmallVector<int64_t, 4> memRefStrides;
      getStridesAndOffset(loadOp.getMemRefType(), memRefStrides, memRefOffset);
      if ((addIOperand == lastIndiceOfLoop) && (memRefStrides.back() == 1))
        toReplace.insert(loadOp);
    }
  }

public:
  explicit BlockOpsImpl(scf::ParallelOp loop) : loop(loop) {}

  void replaceLoadOp(LoadOp op) {
    // Replace with SubgroupBlockWriteINTELOp, update indices to remove
    // subgroup_id, we know at this point that last index comes from the
    // AddI operation, where second operand is subgroup_id
    OpBuilder builder(op);
    SmallVector<Value, 4> newIndices = op.indices();
    auto addOp = newIndices.back().getDefiningOp();
    newIndices.pop_back();
    newIndices.push_back(addOp->getOperand(0));
    auto newBlockReadOp = builder.create<SubgroupBlockReadINTELOp>(
        op.getLoc(), op.memref(), newIndices);
    op.replaceAllUsesWith(newBlockReadOp.getResult());
    op.erase();
  }

  void replaceStoreOp(StoreOp op) {
    // Replace with SubgroupBlockWriteINTELOp, update indice to remove
    // subgroup_id, we know at this point that last indice comes from the
    // addi operation, where second operand is subgroup_id
    OpBuilder builder(op);
    SmallVector<Value, 4> newIndices = op.indices();
    auto addOp = newIndices.back().getDefiningOp();
    newIndices.pop_back();
    newIndices.push_back(addOp->getOperand(0));
    builder.create<SubgroupBlockWriteINTELOp>(op.getLoc(), op.value(),
                                              op.memref(), newIndices);
    op.erase();
  }

  LogicalResult blockOps() {
    mlir::Block *body = loop.getBody();

    for (auto &op : body->getOperations()) {
      // Check if the load/store ops can be replaced with subgroup block ops
      if (auto loadOp = dyn_cast<LoadOp>(op))
        checkLoadStore(loadOp);
      else if (auto storeOp = dyn_cast<StoreOp>(op))
        checkLoadStore(storeOp);
    }

    if (toReplace.empty()) {
      // No valid vector ops used, nothing to do
      return success();
    }

    // Do the Ops transform
    for (auto &op : llvm::make_early_inc_range(body->getOperations())) {
      if (auto loadOp = dyn_cast<LoadOp>(&op))
        replaceLoadOp(loadOp);
      else if (auto storeOp = dyn_cast<StoreOp>(&op))
        replaceStoreOp(storeOp);
    }
    return success();
  }
};

/// Returns true if no other scf.parallel ops are nested within.
static bool isInnermostParallelOp(scf::ParallelOp parallelOp) {
  // Only for the innermost scf.parallel op's.
  bool isInnermost = true;
  parallelOp.walk([&](scf::ParallelOp thisparallelOp) {
    // Since this is a post order walk, we are able to conclude here.
    isInnermost = (thisparallelOp == parallelOp);
    return WalkResult::interrupt();
  });
  return isInnermost;
}

/// Gathers loops that have no scf.parallel nested within.
static void gatherInnermostLoops(FuncOp f,
                                 SmallVectorImpl<scf::ParallelOp> &loops) {
  f.walk([&](scf::ParallelOp parallelOp) {
    if (isInnermostParallelOp(parallelOp))
      loops.push_back(parallelOp);
  });
}

struct SubgroupBlockOpsPass
    : public SubgroupBlockOpsBase<SubgroupBlockOpsPass> {
  void runOnFunction() final {
    auto func = getFunction();
    SmallVector<scf::ParallelOp, 4> loops;
    gatherInnermostLoops(func, loops);
    for (auto parallelOp : loops) {
      BlockOpsImpl impl(parallelOp);
      impl.blockOps();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSubgroupBlockOpsPass() {
  return std::make_unique<SubgroupBlockOpsPass>();
}

} // namespace pmlc::dialect::stdx
