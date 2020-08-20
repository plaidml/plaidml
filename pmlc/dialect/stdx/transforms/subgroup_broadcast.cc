// Copyright 2020 Intel Corporation

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/stdx/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::stdx {

namespace {

/// Finds proper sequence of vector operations, performs unvectorization and
/// adds stdx.subgroup_broadcast
///
/// Sample code:
///
///  %c1_i32 = constant 1 : i32
///  %c0 = constant 0 : index
///  %c8 = constant 8 : index
///  %c64 = constant 64 : index
///  %cst = constant 0.000000e+00 : f32
///  scf.parallel (%i) = (%c0) to (%c64) step (%c8) {
///    %0 = vector.transfer_read %arg0[%i], %cst : memref<64xf32>, vector<8xf32>
///    %1 = vector.extractelement %0[%c1_i32 : i32] : vector<8xf32>
///    %2 = vector.broadcast %1 : f32 to vector<8xf32>
///    vector.transfer_write %2, %arg1[%i] : vector<8xf32>, memref<64xf32>
///    scf.yield
///  }
///
/// Code after transformation:
///
///  %c1_i32 = constant 1 : i32
///  %c0 = constant 0 : index
///  %c1 = constant 1 : index
///  %c8 = constant 8 : index
///  %c64 = constant 64 : index
///  scf.parallel (%i, %sid) = (%c0, %c0) to (%c64, %c8) step (%c8, %c1) {
///    %idx = addi %i, %sid : index
///    %0 = load %arg0[%idx] : memref<64xf32>
///    %1 = stdx.subgroup_broadcast %0, %c1_i32 : f32
///    store %1, %arg1[%idx] : memref<64xf32>
///    scf.yield

void runOnParallel(scf::ParallelOp loopOp) {
  // First, find the matching pattern that needs to have minimum 4 ops within
  // scf.parallel and the following vector ops:
  // transfer_read -> elementextract -> broadcast -> transfer_write
  const unsigned kMinNumValidSubgroupsBroadcastRegion = 4;
  auto *body = loopOp.getBody();
  if (body->getOperations().size() < kMinNumValidSubgroupsBroadcastRegion) {
    IVLOG(5, "The scf.parallel region didn't have the right number of "
             "instructions for a Subgroups broadcast transformation");
    return;
  }

  // TODO: Verify if that is the only case
  if (loopOp.lowerBound().size() != 1) {
    IVLOG(3, "scf.parralel needs to have single dimension");
    return;
  }

  // TODO: transfer_write needs to be in the last place. Verify if this is the
  // only case
  auto it = std::prev(body->end(), 2);
  auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(*it);

  if (!transferWriteOp) {
    IVLOG(3, "The scf.parallel region didn't have a vector.transfer_write as "
             "its last "
             "non-terminator");
    return;
  }

  auto broadcastOp = dyn_cast_or_null<vector::BroadcastOp>(
      transferWriteOp.vector().getDefiningOp());
  if (!broadcastOp) {
    IVLOG(3, "vector() of the vector.transfer_write operation is not from "
             "vector.broadcast");
    return;
  }

  auto extractElementOp = dyn_cast_or_null<vector::ExtractElementOp>(
      broadcastOp.source().getDefiningOp());
  if (!extractElementOp) {
    IVLOG(3, "The source of the vector.broadcast is not vector.extractelement");
    return;
  }

  auto transferReadOp = dyn_cast_or_null<vector::TransferReadOp>(
      extractElementOp.vector().getDefiningOp());
  if (!transferReadOp) {
    IVLOG(3, "vector() of the vector.extractelement operation is not from "
             "vector.transfer_read");
    return;
  }

  mlir::OpBuilder builder(loopOp);
  // Create proper lower,upper bounds and steps for the updated scf.parallel op
  // and create it
  SmallVector<Value, 2> newLowerBounds = loopOp.lowerBound();
  SmallVector<Value, 2> newUpperBounds = loopOp.upperBound();
  SmallVector<Value, 2> newStepsBounds = loopOp.step();
  newLowerBounds.push_back(newLowerBounds[0]);
  newUpperBounds.push_back(newStepsBounds[0]);
  newStepsBounds.push_back(builder.create<ConstantIndexOp>(loopOp.getLoc(), 1));

  auto newLoop = builder.create<scf::ParallelOp>(
      loopOp.getLoc(), newLowerBounds, newUpperBounds, newStepsBounds);

  // Rewrite the new loops body with the proper indices mappings
  BlockAndValueMapping indicesMappings;
  indicesMappings.map(loopOp.getBody()->getArguments(),
                      newLoop.getBody()->getArguments());
  builder.setInsertionPointToStart(newLoop.getBody());
  for (auto &op : loopOp.getBody()->without_terminator())
    builder.clone(op, indicesMappings);

  // Update IVs with newly created index variable
  auto newIVs = newLoop.getInductionVars();
  builder.setInsertionPointToStart(newLoop.getBody());
  auto loc = newLoop.getLoc();
  auto Idx = builder.create<AddIOp>(loc, newIVs[0], newIVs[1]);
  newIVs[0].replaceAllUsesExcept(
      Idx, SmallPtrSet<Operation *, 1>{Idx.getOperation()});

  // Capture the new loop's ops, we already verified ops types before so there
  // is no need to check is the dynamic cast succeeded
  // TODO: again, make sure that transfer_write as the last place will be always
  // the case
  auto *newBody = newLoop.getBody();
  auto newIterator = std::prev(newBody->end(), 2);
  auto newTransferWriteOp = dyn_cast<vector::TransferWriteOp>(*newIterator);
  auto newVecBroadcastOp = dyn_cast_or_null<vector::BroadcastOp>(
      newTransferWriteOp.vector().getDefiningOp());
  auto newExtractElementOp = dyn_cast_or_null<vector::ExtractElementOp>(
      newVecBroadcastOp.source().getDefiningOp());
  auto newTransferReadOp = dyn_cast_or_null<vector::TransferReadOp>(
      newExtractElementOp.vector().getDefiningOp());

  // Create new ops with stdx.subgroup_broadcast
  auto newLoadOp = builder.create<LoadOp>(loc, newTransferReadOp.memref(),
                                          newTransferReadOp.indices());
  auto newBroadcast = builder.create<stdx::SubgroupBroadcastOp>(
      loc, newLoadOp.result().getType(), newLoadOp.result(),
      newExtractElementOp.position());
  builder.create<StoreOp>(loc, newBroadcast.getResult(),
                          newTransferWriteOp.memref(),
                          newTransferWriteOp.indices());

  // Remove old loop and redundant ops from the new loop, replace usages where
  // appropriate
  for (unsigned int i = 0; i < loopOp.getResults().size(); i++)
    loopOp.getResult(i).replaceAllUsesWith(newLoop.getResult(i));
  loopOp.erase();

  newTransferReadOp.vector().replaceAllUsesWith(newLoadOp.result());
  newTransferReadOp.erase();
  newVecBroadcastOp.vector().replaceAllUsesWith(newBroadcast.result());
  newVecBroadcastOp.erase();
  newTransferWriteOp.erase();
  newExtractElementOp.erase();
}

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

struct SubgroupBroadcastPass
    : public SubgroupBroadcastBase<SubgroupBroadcastPass> {
  void runOnFunction() final {
    auto func = getFunction();
    SmallVector<scf::ParallelOp, 4> loops;
    gatherInnermostLoops(func, loops);
    for (auto parallelOp : loops)
      runOnParallel(parallelOp);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSubgroupBroadcastPass() {
  return std::make_unique<SubgroupBroadcastPass>();
}

} // namespace pmlc::dialect::stdx
