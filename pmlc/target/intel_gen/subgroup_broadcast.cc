// Copyright 2020 Intel Corporation

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/GPU/ParallelLoopMapper.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/target/intel_gen/pass_detail.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/tags.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::intel_gen {

namespace {

class DevectorizeImpl {
private:
  scf::ParallelOp loop;
  unsigned vectorSize;
  Value sid;
  bool useBlockOps;

public:
  DevectorizeImpl(scf::ParallelOp loop, unsigned vectorSize, bool useBlockOps)
      : loop(loop), vectorSize(vectorSize), useBlockOps(useBlockOps) {}

  bool isVectorTypeValid(VectorType type) {
    return type.getRank() == 1 && type.getDimSize(0) == vectorSize;
  }

  LogicalResult devectorizeVectorOp(Operation *op) {
    OpBuilder builder(op);
    // Skip on std.extractelement as it will be removed with broadcast op
    // transformation
    if (dyn_cast_or_null<ExtractElementOp>(op))
      return success();

    // Update the result type if it is a vector, operands types should have
    // already been updated before
    for (auto result : op->getResults()) {
      if (auto vtype = result.getType().dyn_cast<VectorType>()) {
        if (!isVectorTypeValid(vtype))
          return failure();
        result.setType(vtype.getElementType());
      }
    }
    return success();
  }

  LogicalResult devectorizeTransferRead(vector::TransferReadOp op) {
    if (!isVectorTypeValid(op.getVectorType()) || op.indices().size() < 1)
      return failure();
    OpBuilder builder(op);
    // Add sid to lowest index
    SmallVector<Value, 4> idxs = op.indices();
    if (useBlockOps) {
      // TODO: add additional requirements like mem alignment
      auto newBlockReadOp =
          builder.create<dialect::stdx::SubgroupBlockReadINTELOp>(
              op.getLoc(), op.memref(), idxs);
      op.replaceAllUsesWith(newBlockReadOp.getResult());
    } else {
      idxs.back() = builder.create<AddIOp>(op.getLoc(), idxs.back(), sid);
      auto newLoadOp = builder.create<LoadOp>(op.getLoc(), op.memref(), idxs);
      op.replaceAllUsesWith(newLoadOp.getResult());
    }
    op.erase();
    return success();
  }

  LogicalResult devectorizeTransferWrite(vector::TransferWriteOp op) {
    if (op.indices().size() < 1)
      return failure();
    OpBuilder builder(op);
    // Add sid to lowest index
    SmallVector<Value, 4> idxs = op.indices();
    if (useBlockOps) {
      // TODO: add additional requirements like mem alignment
      builder.create<dialect::stdx::SubgroupBlockWriteINTELOp>(
          op.getLoc(), op.vector(), op.memref(), idxs);
    } else {
      idxs.back() = builder.create<AddIOp>(op.getLoc(), idxs.back(), sid);
      builder.create<StoreOp>(op.getLoc(), op.vector(), op.memref(), idxs);
    }
    op.erase();
    return success();
  }

  LogicalResult devectorizeAlloc(AllocOp op) {
    auto oldMemRefType = op.getType();
    auto vecType = oldMemRefType.getElementType().dyn_cast<VectorType>();
    if (!vecType) {
      // Ignore non-vector allocates
      return success();
    }
    auto shape = vecType.getShape();
    if (shape.size() != 1 || shape[0] != vectorSize) {
      // Vector allocations must match the subgroup size
      return failure();
    }
    auto newMemRefType = MemRefType::Builder(oldMemRefType)
                             .setElementType(vecType.getElementType());
    op.getResult().setType(newMemRefType);
    return success();
  }

  LogicalResult devectorizeBroadcast(vector::BroadcastOp op) {
    // If broadcast's result comes from extractelement then replace it with
    // stdx.subgroup_broadcast, otherwise it is removed
    if (!isVectorTypeValid(op.getVectorType()))
      return failure();
    auto extractElementOp =
        dyn_cast_or_null<ExtractElementOp>(op.source().getDefiningOp());
    if (extractElementOp) {
      OpBuilder builder(op);
      auto newBroadcast = builder.create<dialect::stdx::SubgroupBroadcastOp>(
          op.getLoc(), extractElementOp.aggregate().getType(),
          extractElementOp.aggregate(), extractElementOp.indices()[0]);
      op.replaceAllUsesWith(newBroadcast.getResult());
      extractElementOp.replaceAllUsesWith(newBroadcast.getResult());
      extractElementOp.erase();
      op.erase();
    } else {
      op.replaceAllUsesWith(op.source());
      op.erase();
    }
    return success();
  }

  LogicalResult devectorizeOperation(Operation *op) {
    if (auto vecReadOp = dyn_cast<vector::TransferReadOp>(op)) {
      return devectorizeTransferRead(vecReadOp);
    } else if (auto vecWriteOp = dyn_cast<vector::TransferWriteOp>(op)) {
      return devectorizeTransferWrite(vecWriteOp);
    } else if (auto vecBroadcastOp = dyn_cast<vector::BroadcastOp>(op)) {
      return devectorizeBroadcast(vecBroadcastOp);
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      for (auto &iop :
           llvm::make_early_inc_range(forOp.getBody()->getOperations())) {
        if (failed(devectorizeOperation(&iop))) {
          return failure();
        }
      }
      return success();
    } else if (auto allocOp = dyn_cast<AllocOp>(op)) {
      return devectorizeAlloc(allocOp);
    } else {
      return devectorizeVectorOp(op);
    }
  }

  LogicalResult devectorize() {
    mlir::Block *body = loop.getBody();

    // Rewrite the loop with new dimension added
    mlir::OpBuilder builder(loop);

    // Create proper lower, upper bounds and steps for the updated
    // scf.parallel op and create the actual op
    SmallVector<Value, 4> newLowerBounds = loop.lowerBound();
    SmallVector<Value, 4> newUpperBounds = loop.upperBound();
    SmallVector<Value, 4> newStepsBounds = loop.step();
    auto loc = loop.getLoc();

    // Add an initial index for the sid, with a range of vectorSize.
    newLowerBounds.insert(newLowerBounds.begin(),
                          builder.create<ConstantIndexOp>(loc, 0));
    newUpperBounds.insert(newUpperBounds.begin(),
                          builder.create<ConstantIndexOp>(loc, vectorSize));
    newStepsBounds.insert(newStepsBounds.begin(),
                          builder.create<ConstantIndexOp>(loc, 1));

    auto newLoop = builder.create<scf::ParallelOp>(
        loop.getLoc(), newLowerBounds, newUpperBounds, newStepsBounds);
    mlir::Block *newBody = newLoop.getBody();

    // Splice across interior + erase orig
    auto &oldBodyOps = body->getOperations();
    auto &newBodyOps = newBody->getOperations();
    newBodyOps.splice(std::prev(newBodyOps.end()), oldBodyOps,
                      oldBodyOps.begin(), std::prev(oldBodyOps.end()));

    // Replace block argsA
    for (size_t i = 0; i < body->getNumArguments(); i++) {
      body->getArgument(i).replaceAllUsesWith(newBody->getArgument(i + 1));
    }
    sid = newBody->getArgument(0);

    // Do the Ops transform
    for (auto &op : llvm::make_early_inc_range(newBody->getOperations())) {
      if (failed(devectorizeOperation(&op))) {
        return failure();
      }
    }

    // Reset GPU thread mappings
    SmallVector<gpu::ParallelLoopDimMapping, 8> mappings;
    auto proc = gpu::Processor::ThreadX;
    for (unsigned i = 0; i < newBody->getNumArguments(); i++) {
      mappings.push_back(getParallelLoopDimMappingAttr(
          proc, builder.getDimIdentityMap(), builder.getDimIdentityMap()));
      proc = static_cast<gpu::Processor>(static_cast<int>(proc) + 1);
    }
    setMappingAttr(newLoop, mappings);
    loop.erase();
    return success();
  }
};

struct SubgroupBroadcastPass
    : public SubgroupBroadcastBase<SubgroupBroadcastPass> {
  SubgroupBroadcastPass() = default;
  explicit SubgroupBroadcastPass(bool useBlockOps) {
    this->useBlockOps = useBlockOps;
  }
  void runOnFunction() final {
    auto func = getFunction();
    IVLOG(1, "Doing something");
    func.walk([&](scf::ParallelOp op) {
      int64_t subgroupSize = getIntegerTag(op, subgroupSizeTag(), 1);
      if (hasUnitTag(op, gpuThreadTag()) && subgroupSize > 1) {
        DevectorizeImpl impl(op, subgroupSize, useBlockOps.getValue());
        if (failed(impl.devectorize())) {
          signalPassFailure();
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSubgroupBroadcastPass() {
  return std::make_unique<SubgroupBroadcastPass>();
}

std::unique_ptr<mlir::Pass> createSubgroupBroadcastPass(bool useBlockOps) {
  return std::make_unique<SubgroupBroadcastPass>(useBlockOps);
}

} // namespace pmlc::target::intel_gen
