// Copyright 2020 Intel Corporation

#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

#include "pmlc/dialect/comp/analysis/mem_sync_tracker.h"
#include "pmlc/dialect/comp/ir/dialect.h"
#include "pmlc/dialect/comp/transforms/pass_detail.h"
#include "pmlc/dialect/comp/transforms/passes.h"

namespace pmlc::dialect::comp {

namespace {

class RemoveRedundantRWPass final
    : public RemoveRedundantRWBase<RemoveRedundantRWPass> {
public:
  void runOnFunction();

  void removeTransfer(MemoryTransferOpInterface op);
};

void RemoveRedundantRWPass::runOnFunction() {
  mlir::FuncOp func = getFunction();
  std::vector<MemoryTransferOpInterface> removeList;

  for (mlir::Block &block : func) {
    MemorySynchronizationTracker tracker;

    block.walk([&](mlir::Operation *operation) {
      bool changed = tracker.handleOperation(operation);
      auto transferOp = mlir::dyn_cast<MemoryTransferOpInterface>(operation);
      if (!changed && transferOp)
        removeList.push_back(transferOp);
    });
  }
  for (MemoryTransferOpInterface toRemove : removeList)
    removeTransfer(toRemove);
}

void RemoveRedundantRWPass::removeTransfer(MemoryTransferOpInterface op) {
  mlir::Operation *operation = op.getOperation();
  if (auto scheduleTransferOp =
          mlir::dyn_cast<ScheduleOpInterface>(operation)) {
    mlir::Value resulting = scheduleTransferOp.getResultingEvent();
    mlir::OperandRange removedDependencies =
        scheduleTransferOp.getDependencies();

    if (removedDependencies.size() == 1) {
      operation->replaceAllUsesWith(removedDependencies);
      operation->erase();
      return;
    }

    auto replaceDependency = [&](mlir::ValueRange oldDependencies,
                                 mlir::MutableOperandRange operands) {
      mlir::SmallVector<mlir::Value, 4> newDependencies;
      for (mlir::Value dep : oldDependencies)
        if (dep != resulting)
          newDependencies.push_back(dep);
        else
          newDependencies.append(removedDependencies.begin(),
                                 removedDependencies.end());
      operands.assign(newDependencies);
    };

    for (mlir::Operation *user :
         llvm::make_early_inc_range(operation->getUsers())) {
      if (auto scheduleUser = mlir::dyn_cast<ScheduleOpInterface>(user)) {
        replaceDependency(scheduleUser.getDependencies(),
                          scheduleUser.getDependencyMutable());
      } else if (auto wait = mlir::dyn_cast<Wait>(user)) {
        replaceDependency(wait.events(), wait.eventsMutable());
        if (wait.events().empty())
          user->erase();
      } else {
        scheduleTransferOp
                .emitRemark("could not remove redundant operation - unknown "
                            "replacement semantic")
                .attachNote(user->getLoc())
            << "see user: " << *user;
      }
    }
  }

  if (operation->use_empty())
    operation->erase();
}

} // namespace

std::unique_ptr<mlir::Pass> createRemoveRedundantRWPass() {
  return std::make_unique<RemoveRedundantRWPass>();
}

} // namespace pmlc::dialect::comp
