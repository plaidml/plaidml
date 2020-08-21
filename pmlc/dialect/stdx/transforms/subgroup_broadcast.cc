// Copyright 2020 Intel Corporation

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/stdx/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::stdx {

namespace {

/// Performs unvectorization and finds the sequence of
/// vector.extractelement->vector.broadcast and replaces with
/// stdx.subgroup_broadcast
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

class DevectorizeImpl {
private:
  scf::ParallelOp loop;
  unsigned vectorSize;
  DenseSet<Value> vectorizedValues;
  DenseSet<Operation *> vectorizedOps;

  LogicalResult tryDevectorizeOperation(Operation *op) {
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        // For each Vector operation check if vector is single ranked and of
        // size of parallel's loop step on last dimension (vectorSize)
        .Case<vector::TransferReadOp>([&](auto op) {
          auto vecType = op.vector().getType().template cast<VectorType>();
          if ((vecType.getRank() != 1) ||
              (vecType.getDimSize(0) != vectorSize)) {
            IVLOG(3, "Devectorize: Failed, TransferReadOp vector size does not "
                     "match last dimension step of the loop");
            return failure();
          }
          vectorizedOps.insert(op);
          vectorizedValues.insert(op.getResult());
          return success();
        })
        .Case<vector::TransferWriteOp>([&](auto op) {
          auto vecType = op.vector().getType().template cast<VectorType>();
          if ((vecType.getRank() != 1) ||
              (vecType.getDimSize(0) != vectorSize)) {
            IVLOG(3, "Devectorize: Failed, TransferReadOp vector size does not "
                     "match last dimension step of the loop");
            return failure();
          }
          vectorizedOps.insert(op);
          return success();
        })
        .Case<vector::BroadcastOp>([&](auto op) {
          auto vecType = op.vector().getType().template cast<VectorType>();
          if ((vecType.getRank() != 1) ||
              (vecType.getDimSize(0) != vectorSize)) {
            IVLOG(3, "Devectorize: Failed, BroadcastOp vector size does not "
                     "match last dimension step of the loop");
            return failure();
          }
          vectorizedOps.insert(op);
          vectorizedValues.insert(op.getResult());
          return success();
        })
        .Default([&](Operation *op) {
          // Check if the operation is vectorizable to consider it at all with
          // transformation
          if (!mlir::isa<VectorUnrollOpInterface>(op)) {
            // Create exception for vector.extractelement followed by
            // vector.broadcast that will be taken care of later.
            auto extractElementOp =
                dyn_cast_or_null<vector::ExtractElementOp>(op);
            if (extractElementOp) {
              for (auto user : extractElementOp.result().getUsers()) {
                if (dyn_cast_or_null<vector::BroadcastOp>(user))
                  return success();
                else
                  break;
              }
            }
            return success();
          }
          // Devectorize if at least one operand will be devectorized
          bool anyDevec = false;
          for (auto operand : op->getOperands()) {
            if (vectorizedValues.count(operand)) {
              anyDevec = true;
            }
          }
          if (!anyDevec) {
            // No need to devectorize, all is good
            return success();
          }
          // Don't handle ops with multiple results
          if (op->getNumResults() != 1) {
            IVLOG(3, "Devectorize: Failed, multi-result vector op");
            return failure();
          }
          vectorizedOps.insert(op);
          vectorizedValues.insert(op->getResult(0));
          return success();
        });
  }

public:
  explicit DevectorizeImpl(scf::ParallelOp loop) : loop(loop) {}

  void devectorizeVectorOp(Operation *op) {
    OpBuilder builder(op);
    // Skip on vector.extractelement as it will be removed with broadcast op
    // transformation
    if (dyn_cast_or_null<vector::ExtractElementOp>(op))
      return;

    // Update the result type if it is a vector, operands types should have
    // already been updated before
    for (auto result : op->getResults()) {
      auto resultType = result.getType();
      if (resultType.isa<VectorType>())
        result.setType(resultType.template cast<VectorType>().getElementType());
    }
  }

  void devectorizeTransferRead(vector::TransferReadOp op) {
    // Simple replace with loadOp, the indices were already updated before
    OpBuilder builder(op);
    auto newLoadOp =
        builder.create<LoadOp>(op.getLoc(), op.memref(), op.indices());
    op.replaceAllUsesWith(newLoadOp.getResult());
    op.erase();
  }

  void devectorizeTransferWrite(vector::TransferWriteOp op) {
    // Simple replace with storeOp, the indices were already updated before
    OpBuilder builder(op);
    builder.create<StoreOp>(op.getLoc(), op.vector(), op.memref(),
                            op.indices());
    op.erase();
  }

  void devectorizeBroadcast(vector::BroadcastOp op) {
    // If broadcast's result comes from extractelement then replace it with
    // stdx.subgroup_broadcast, otherwise it is removed
    auto extractElementOp =
        dyn_cast_or_null<vector::ExtractElementOp>(op.source().getDefiningOp());
    if (extractElementOp) {
      OpBuilder builder(op);
      auto newBroadcast = builder.create<stdx::SubgroupBroadcastOp>(
          op.getLoc(), extractElementOp.vector().getType(),
          extractElementOp.vector(), extractElementOp.position());
      op.replaceAllUsesWith(newBroadcast.getResult());
      extractElementOp.replaceAllUsesWith(newBroadcast.getResult());
      extractElementOp.erase();
      op.erase();
    } else {
      op.replaceAllUsesWith(op.source());
      op.erase();
    }
  }

  void devectorizeOperation(Operation *op) {
    if (auto vecReadOp = dyn_cast<vector::TransferReadOp>(op)) {
      devectorizeTransferRead(vecReadOp);
    } else if (auto vecWriteOp = dyn_cast<vector::TransferWriteOp>(op)) {
      devectorizeTransferWrite(vecWriteOp);
    } else if (auto vecBroadcastOp = dyn_cast<vector::BroadcastOp>(op)) {
      devectorizeBroadcast(vecBroadcastOp);
    } else {
      devectorizeVectorOp(op);
    }
  }

  LogicalResult devectorize() {
    mlir::Block *body = loop.getBody();

    // TODO: Check if setting vector size by the step of the last dimension
    // will be ok for our implementation
    auto vectorSizeOp =
        dyn_cast_or_null<ConstantOp>(loop.step().back().getDefiningOp());
    if (!vectorSizeOp) {
      IVLOG(3, "Devectorize: Failed to get step size");
      return failure();
    }

    vectorSize = vectorSizeOp.value().template cast<IntegerAttr>().getInt();
    if (vectorSize == 1) {
      // Step on last dimension is 1, no need to devectorize then
      return failure();
    }

    bool devectorizable = true;
    for (auto &op : body->getOperations()) {
      devectorizable &= succeeded(tryDevectorizeOperation(&op));
    }
    if (!devectorizable) {
      return failure();
    }

    if (vectorizedOps.empty()) {
      // No valid vector ops used, nothing to do
      return success();
    }

    // Rewrite the loop with new dimension added
    mlir::OpBuilder builder(loop);

    // Create proper lower, upper bounds and steps for the updated
    // scf.parallel op and create the actual op
    SmallVector<Value, 4> newLowerBounds = loop.lowerBound();
    SmallVector<Value, 4> newUpperBounds = loop.upperBound();
    SmallVector<Value, 4> newStepsBounds = loop.step();
    newLowerBounds.push_back(newLowerBounds.back());
    newUpperBounds.push_back(newStepsBounds.back());
    newStepsBounds.push_back(builder.create<ConstantIndexOp>(loop.getLoc(), 1));

    auto newLoop = builder.create<scf::ParallelOp>(
        loop.getLoc(), newLowerBounds, newUpperBounds, newStepsBounds);

    // Rewrite the new loops body with the proper indices mappings
    BlockAndValueMapping indicesMappings;
    indicesMappings.map(loop.getBody()->getArguments(),
                        newLoop.getBody()->getArguments());
    builder.setInsertionPointToStart(newLoop.getBody());
    for (auto &op : loop.getBody()->without_terminator())
      builder.clone(op, indicesMappings);

    // Update IVs with newly created index variable
    auto newIVs = newLoop.getInductionVars();
    builder.setInsertionPointToStart(newLoop.getBody());
    auto loc = newLoop.getLoc();
    auto idx =
        builder.create<AddIOp>(loc, newIVs[newIVs.size() - 2], newIVs.back());
    newIVs[newIVs.size() - 2].replaceAllUsesExcept(
        idx, SmallPtrSet<Operation *, 1>{idx.getOperation()});

    // Remove old loop and redundant ops from the new loop, replace usages
    // where appropriate
    for (unsigned int i = 0; i < loop.getResults().size(); i++)
      loop.getResult(i).replaceAllUsesWith(newLoop.getResult(i));
    loop.erase();

    mlir::Block *newBody = newLoop.getBody();
    // Do the Ops transform
    for (auto &op : llvm::make_early_inc_range(newBody->getOperations())) {
      devectorizeOperation(&op);
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

struct SubgroupBroadcastPass
    : public SubgroupBroadcastBase<SubgroupBroadcastPass> {
  void runOnFunction() final {
    auto func = getFunction();
    SmallVector<scf::ParallelOp, 4> loops;
    gatherInnermostLoops(func, loops);
    for (auto parallelOp : loops) {
      DevectorizeImpl impl(parallelOp);
      impl.devectorize();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSubgroupBroadcastPass() {
  return std::make_unique<SubgroupBroadcastPass>();
}

} // namespace pmlc::dialect::stdx
