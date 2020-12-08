// Copyright 2020 Intel Corporation

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"

#include "pmlc/dialect/comp/ir/dialect.h"
#include "pmlc/dialect/comp/transforms/pass_detail.h"
#include "pmlc/dialect/comp/transforms/passes.h"

#include "pmlc/dialect/comp/transforms/enums.h.inc"

namespace pmlc::dialect::comp {

namespace {
using pmlc::dialect::comp::BufferCopyMode;

class MinimizeBufferTransfersPass final
    : public MinimizeBufferTransfersBase<MinimizeBufferTransfersPass> {
public:
  void runOnFunction();

private:
  llvm::SmallSet<mlir::Operation *, 4>
  getExternalDependentOperations(mlir::Value value);
  bool isInternalOperation(mlir::Operation *op);
  BufferCopyMode getBufferCopyMode(mlir::Operation *op, mlir::Value buffer);
  void removeWaitOpEvent(comp::Wait waitOp, mlir::Value event);
};

llvm::SmallSet<mlir::Operation *, 4>
MinimizeBufferTransfersPass::getExternalDependentOperations(mlir::Value value) {
  llvm::SmallSet<mlir::Operation *, 4> operations;
  llvm::SetVector<mlir::OpOperand *> uses;

  operations.insert(value.getDefiningOp());
  for (auto &use : value.getUses()) {
    uses.insert(&use);
  }
  while (!uses.empty()) {
    auto owner = uses.back()->getOwner();
    operations.insert(owner);
    uses.pop_back();
    for (auto opResult : owner->getOpResults()) {
      for (auto &use : opResult.getUses()) {
        uses.insert(&use);
      }
    }
  }

  for (auto operation : operations) {
    if (isInternalOperation(operation)) {
      operations.erase(operation);
    }
  }
  return operations;
}

bool MinimizeBufferTransfersPass::isInternalOperation(mlir::Operation *op) {
  auto allocOp = llvm::dyn_cast<mlir::AllocOp>(op);
  auto memRefCastOp = llvm::dyn_cast<mlir::MemRefCastOp>(op);
  auto waitOp = llvm::dyn_cast<comp::Wait>(op);
  if (allocOp || memRefCastOp || waitOp) {
    return true;
  }
  return false;
}

BufferCopyMode
MinimizeBufferTransfersPass::getBufferCopyMode(mlir::Operation *op,
                                               mlir::Value buffer) {
  BufferCopyMode copyMode = BufferCopyMode::NoCopy;
  if (buffer.isa<mlir::BlockArgument>()) {
    copyMode = copyMode |
               (BufferCopyMode::HostToDevice | BufferCopyMode::DeviceToHost);
    return copyMode;
  }

  auto currentBlock = op->getBlock();
  auto operationDeps = getExternalDependentOperations(buffer);
  for (auto dependency : operationDeps) {
    if (dependency->getBlock() == currentBlock) {
      if (dependency->isBeforeInBlock(op)) {
        copyMode = copyMode | BufferCopyMode::HostToDevice;
      }
      if (op->isBeforeInBlock(dependency)) {
        copyMode = copyMode | BufferCopyMode::DeviceToHost;
      }
    } else {
      // Dependency is outside of current block.
      copyMode = copyMode |
                 (BufferCopyMode::HostToDevice | BufferCopyMode::DeviceToHost);
    }
  }
  return copyMode;
}

void MinimizeBufferTransfersPass::removeWaitOpEvent(comp::Wait waitOp,
                                                    mlir::Value event) {
  auto events = waitOp.events();
  for (size_t i = 0; i < events.size(); i++) {
    auto pEvent = std::next(events.begin(), i);
    if (*pEvent == event) {
      waitOp.eventsMutable().erase(i);
      break;
    }
  }
}

void MinimizeBufferTransfersPass::runOnFunction() {
  auto func = getFunction();

  func.walk([&](comp::ScheduleWrite op) {
    auto copyMode = getBufferCopyMode(op.getOperation(), op.hostMem());
    if ((copyMode & BufferCopyMode::HostToDevice) == BufferCopyMode::NoCopy) {
      auto users = op.getResult().getUsers();
      for (auto user : users) {
        if (auto waitOp = mlir::dyn_cast<comp::Wait>(user)) {
          removeWaitOpEvent(waitOp, op.getResult());
        }
      }
      op.replaceAllUsesWith(op.deviceMem());
      op.erase();
    }
  });

  func.walk([&](comp::ScheduleRead op) {
    auto copyMode = getBufferCopyMode(op.getOperation(), op.hostMem());
    if ((copyMode & BufferCopyMode::DeviceToHost) == BufferCopyMode::NoCopy) {
      auto users = op.getResult().getUsers();
      for (auto user : users) {
        if (auto waitOp = mlir::dyn_cast<comp::Wait>(user)) {
          removeWaitOpEvent(waitOp, op.getResult());
        }
      }
      op.erase();
    }
  });

  func.walk([&](comp::Wait op) {
    if (op.events().size() == 0) {
      op.erase();
    }
  });
}

} // namespace

std::unique_ptr<mlir::Pass> createMinimizeBufferTransfersPass() {
  return std::make_unique<MinimizeBufferTransfersPass>();
}

} // namespace pmlc::dialect::comp
