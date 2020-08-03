// Copyright 2020 Intel Corporation
#include <iostream>
#include <map>
#include <memory>

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"

#include "pmlc/dialect/comp/transforms/pass_detail.h"
#include "pmlc/dialect/comp/transforms/passes.h"

namespace pmlc::dialect::comp {

namespace gpu = mlir::gpu;

namespace {

void reuseExecEnv(mlir::FuncOp funcOp) {
  // Finds earliest creation of exec environment with each type and latest
  // destruction. Reuse environment from first creation and remove all
  // destructions except last.
  // TODO This may fail if there are multiple blocks
  llvm::DenseMap<mlir::Type, CreateExecEnv> earliestCreation;
  llvm::DenseMap<mlir::Type, DestroyExecEnv> latestDestruction;

  auto findCreateDestroyFn = [&](mlir::Operation *op) -> void {
    if (auto createOp = mlir::dyn_cast<CreateExecEnv>(op)) {
      auto execEnv = createOp.getResult();
      auto execEnvType = execEnv.getType();
      if (earliestCreation.count(execEnvType) == 0) {
        earliestCreation[execEnvType] = createOp;
      }
    }
    if (auto destroyOp = mlir::dyn_cast<DestroyExecEnv>(op)) {
      auto execEnv = destroyOp.execEnv();
      auto execEnvType = execEnv.getType();
      latestDestruction[execEnvType] = destroyOp;
    }
  };

  auto replaceCreateDestroyFn = [&](mlir::Operation *op) -> void {
    if (auto createOp = mlir::dyn_cast<CreateExecEnv>(op)) {
      auto execEnv = createOp.getResult();
      auto execEnvType = execEnv.getType();
      if (earliestCreation[execEnvType] != createOp) {
        op->replaceAllUsesWith(earliestCreation[execEnvType]);
        op->erase();
      }
      return;
    }
    if (auto destroyOp = mlir::dyn_cast<DestroyExecEnv>(op)) {
      auto execEnv = destroyOp.execEnv();
      auto execEnvType = execEnv.getType();

      if (latestDestruction[execEnvType] != destroyOp) {
        op->erase();
      }
      return;
    }
  };

  funcOp.walk(findCreateDestroyFn);
  funcOp.walk(replaceCreateDestroyFn);
}

void reuseMemory(mlir::FuncOp funcOp) {
  struct AllocationInfo {
    mlir::Value execEnv;
    Alloc allocOp;
    Dealloc deallocOp;
  };

  llvm::DenseMap<mlir::Value, AllocationInfo> deallocatedMemory;
  std::list<std::pair<mlir::Operation *, AllocationInfo>> reusePossibilities;

  auto findReusePossibilitesFn = [&](mlir::Operation *op) -> void {
    if (auto deallocOp = mlir::dyn_cast<Dealloc>(op)) {
      auto deviceMem = deallocOp.deviceMem();
      auto allocOp = mlir::cast<Alloc>(deviceMem.getDefiningOp());
      auto hostMem = allocOp.hostMem();
      if (!hostMem)
        return;
      auto execEnv = allocOp.execEnv();
      deallocatedMemory[hostMem] = AllocationInfo{execEnv, allocOp, deallocOp};
    }
    if (auto allocOp = mlir::dyn_cast<Alloc>(op)) {
      auto hostMem = allocOp.hostMem();
      if (!hostMem || deallocatedMemory.count(hostMem) == 0)
        return;
      auto execEnv = allocOp.execEnv();
      auto deallocated = deallocatedMemory[hostMem];
      if (deallocated.execEnv != execEnv)
        return;
      reusePossibilities.emplace_front(op, deallocated);
      deallocatedMemory.erase(hostMem);
    }
  };

  funcOp.walk(findReusePossibilitesFn);

  for (auto &reuse : reusePossibilities) {
    auto &op = reuse.first;
    auto &allocOp = reuse.second.allocOp;
    auto hostMem = allocOp.hostMem();
    auto deviceMem = allocOp.getResult();
    auto execEnv = reuse.second.execEnv;
    mlir::OpBuilder builder(op);
    // Insert write and wait to make sure that memory is up to data
    auto eventType = builder.getType<EventType>(
        execEnv.getType().cast<ExecEnvType>().getRuntime());
    auto writeOp =
        builder.create<ScheduleWrite>(op->getLoc(), eventType, hostMem,
                                      deviceMem, reuse.second.execEnv, Value());
    auto waitOp = builder.create<Wait>(op->getLoc(), writeOp.getResult());

    op->replaceAllUsesWith(reuse.second.allocOp);
    reuse.second.deallocOp.erase();
    op->erase();
  }
}

mlir::LogicalResult eraseEventDependencies(mlir::FuncOp funcOp) {
  // Check that dependencies can be safely removed
  // Check that schedule operations have known event dependency and dependees
  auto checkSchedule =
      funcOp.walk([&](ScheduleOpInterface op) -> mlir::WalkResult {
        auto outEvent = op.getResultingEvent();
        for (auto user : outEvent.getUsers()) {
          if (!mlir::isa<Wait>(user) && !mlir::isa<GroupEvents>(user) &&
              !mlir::isa<ScheduleOpInterface>(user))
            return mlir::WalkResult::interrupt();
        }
        auto depEvent = op.getDependency();
        if (depEvent) {
          auto definingOp = depEvent.getDefiningOp();
          if (!definingOp || (!mlir::isa<GroupEvents>(definingOp) &&
                              !mlir::isa<ScheduleOpInterface>(definingOp)))
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
  if (checkSchedule.wasInterrupted())
    return mlir::failure();
  // All defining operations for waits are known
  auto checkWait = funcOp.walk([&](Wait op) -> mlir::WalkResult {
    auto depEvent = op.event();
    auto definingOp = depEvent.getDefiningOp();
    return mlir::success(definingOp &&
                         (mlir::isa<ScheduleOpInterface>(definingOp) ||
                          mlir::isa<GroupEvents>(definingOp)));
  });
  if (checkSchedule.wasInterrupted())
    return mlir::failure();
  // All defining operations for group events are known
  auto checkGroup = funcOp.walk([&](GroupEvents op) -> mlir::WalkResult {
    for (auto user : op.getResult().getUsers()) {
      if (!mlir::isa<Wait>(user) && !mlir::isa<GroupEvents>(user) &&
          !mlir::isa<ScheduleOpInterface>(user))
        return mlir::WalkResult::interrupt();
    }
    for (auto ev : op.events()) {
      auto definingOp = ev.getDefiningOp();
      if (!definingOp || (!mlir::isa<ScheduleOpInterface>(definingOp) &&
                          !mlir::isa<GroupEvents>(definingOp)))
        return mlir::WalkResult::interrupt();
    }

    return mlir::WalkResult::advance();
  });
  if (checkGroup.wasInterrupted())
    return mlir::failure();

  // Erase all waits
  funcOp.walk([&](Wait op) { op.erase(); });
  // Erase all dependency information from schedule operations
  funcOp.walk(
      [&](ScheduleOpInterface op) { op.getDependencyMutable().clear(); });
  // Erase all group events
  bool allGroupErased = false;
  while (!allGroupErased) {
    allGroupErased = true;
    bool atLeastOneErased = false;
    funcOp.walk([&](GroupEvents op) {
      if (op.use_empty()) {
        atLeastOneErased = true;
        op.erase();
      } else {
        allGroupErased = false;
      }
    });
    if (!allGroupErased && !atLeastOneErased) {
      // Something went terribly wrong, there is something holding the use of
      // event that wasn't discovered in first checks. This shouldn't happen,
      // but just for safety of not having infinite loop.
      return mlir::failure();
    }
  }

  return mlir::success();
}

void removeRedundantRW(mlir::FuncOp funcOp) {
  mlir::DenseMap<mlir::Value, std::list<mlir::Operation *>> allOperations;
  mlir::DenseMap<mlir::Value, mlir::DenseSet<mlir::Value>> possibleAliases;

  auto isUsingAlias = [&](mlir::Operation *op, mlir::Value val) -> bool {
    for (auto operand : op->getOperands()) {
      if (operand == val)
        return true;
      for (auto alias : possibleAliases[val]) {
        if (operand == alias)
          return true;
      }
    }
    return false;
  };

  auto fillAliases = [&](mlir::Operation *op) -> void {
    for (auto aliased : possibleAliases) {
      if (isUsingAlias(op, aliased.first))
        allOperations[aliased.first].push_back(op);
    }
  };

  auto gatherReadWriteOps = [&](mlir::Operation *op) -> void {
    if (auto allocOp = mlir::dyn_cast<Alloc>(op)) {
      auto deviceMem = allocOp.getResult();
      allOperations[deviceMem] = {};
      possibleAliases[deviceMem] = {};
      auto hostMem = allocOp.hostMem();
      if (hostMem)
        possibleAliases[deviceMem].insert(hostMem);
      allOperations[deviceMem].push_back(op);
    } else if (auto readOp = mlir::dyn_cast<ScheduleRead>(op)) {
      auto deviceMem = readOp.deviceMem();
      auto hostMem = readOp.hostMem();
      possibleAliases[deviceMem].insert(hostMem);
      allOperations[deviceMem].push_back(op);
    } else if (auto writeOp = mlir::dyn_cast<ScheduleWrite>(op)) {
      auto deviceMem = writeOp.deviceMem();
      auto hostMem = writeOp.hostMem();
      possibleAliases[deviceMem].insert(hostMem);
      allOperations[deviceMem].push_back(op);
    } else {
      fillAliases(op);
    }
  };

  funcOp.walk(gatherReadWriteOps);

  llvm::DenseSet<mlir::Operation *> toRemove;
  // Analyse what can be removed
  for (auto valUsers : allOperations) {
    auto deviceMem = valUsers.first;
    mlir::DenseSet<mlir::Value> synchronizedHostMem;
    for (auto user : valUsers.second) {
      if (auto allocOp = mlir::dyn_cast<Alloc>(user)) {
        auto hostMem = allocOp.hostMem();
        if (hostMem)
          synchronizedHostMem.insert(hostMem);
      } else if (auto readOp = mlir::dyn_cast<ScheduleRead>(user)) {
        auto hostMem = readOp.hostMem();

        if (synchronizedHostMem.count(hostMem))
          toRemove.insert(user);
        else
          synchronizedHostMem.insert(hostMem);
      } else if (auto writeOp = mlir::dyn_cast<ScheduleWrite>(user)) {
        auto hostMem = writeOp.hostMem();

        if (synchronizedHostMem.count(hostMem)) {
          toRemove.insert(user);
        } else {
          synchronizedHostMem.clear();
          synchronizedHostMem.insert(hostMem);
        }
      } else {
        for (auto operand : user->getOperands()) {
          if (operand == deviceMem) {
            synchronizedHostMem.clear();
            break;
          } else if (synchronizedHostMem.count(operand)) {
            synchronizedHostMem.erase(operand);
          }
        }
      }
    }
  }

  // Find read -> (...) -> read pattern with (...) not using device memory
  for (auto valUsers : allOperations) {
    llvm::DenseSet<mlir::Value> unusedHost;
    llvm::DenseMap<mlir::Value, mlir::Operation *> previousRead;
    for (auto user : valUsers.second) {
      if (toRemove.count(user))
        continue;
      if (auto readOp = mlir::dyn_cast<ScheduleRead>(user)) {
        auto hostMem = readOp.hostMem();

        if (unusedHost.count(hostMem)) {
          toRemove.insert(previousRead[hostMem]);
        } else {
          unusedHost.insert(hostMem);
          previousRead[hostMem] = user;
        }
      } else {
        for (auto operand : user->getOperands()) {
          if (unusedHost.count(operand)) {
            unusedHost.erase(operand);
            previousRead.erase(operand);
          }
        }
      }
    }
  }

  // Remove
  for (auto op : toRemove) {
    op->erase();
  }
}

void recalculateEventDependencies(mlir::FuncOp funcOp) {
  // Recreate waits by analysing memory dependencies
  llvm::DenseMap<mlir::Value, mlir::Value> memoryEventMap;

  auto gatherEventDependencies = [&](mlir::Operation *op) {
    // Skip gpu.launch_func as it will be handled by parent comp.schedule_func
    if (mlir::isa<gpu::LaunchFuncOp>(op))
      return;

    mlir::OpBuilder builder(op);
    std::vector<mlir::Value> dependencies;
    std::vector<mlir::Value> dependees;

    if (auto scheduleFuncOp = mlir::dyn_cast<ScheduleFunc>(op)) {
      // Special case for schedule_func, as its dependencies should be taken
      // from actual function launch
      auto launchOp =
          mlir::cast<gpu::LaunchFuncOp>(scheduleFuncOp.body().front().front());
      for (auto &operand : launchOp.getOperation()->getOperands()) {
        if (memoryEventMap.count(operand)) {
          dependencies.push_back(memoryEventMap[operand]);
          dependees.push_back(operand);
        } else if (operand.getType().isa<mlir::MemRefType>()) {
          dependees.push_back(operand);
        }
      }
    } else {
      for (auto operand : op->getOperands()) {
        if (memoryEventMap.count(operand)) {
          dependencies.push_back(memoryEventMap[operand]);
          dependees.push_back(operand);
        } else if (operand.getType().isa<mlir::MemRefType>()) {
          dependees.push_back(operand);
        }
      }
    }

    if (auto scheduleOp = mlir::dyn_cast<ScheduleOpInterface>(op)) {
      // For operations with schedule interface dependencies become operation
      // parameter
      if (!dependencies.empty()) {
        auto execEnvOp = mlir::cast<ExecEnvOpInterface>(op);
        auto execEnvType = execEnvOp.getExecEnv().getType().cast<ExecEnvType>();
        auto eventType = builder.getType<EventType>(execEnvType.getRuntime());
        auto groupOp = builder.createOrFold<GroupEvents>(
            op->getLoc(), eventType, dependencies);
        scheduleOp.getDependencyMutable().assign(groupOp);
      }
      // Update dependees with resulting event
      for (auto dep : dependees) {
        memoryEventMap[dep] = scheduleOp.getResultingEvent();
      }
    } else {
      // For non-schedule operations waits are needed to ensure correct
      // ordering. Separate wait for each environment.
      llvm::DenseMap<mlir::Type, std::vector<mlir::Value>> eventTypeGroups;
      for (auto dep : dependencies) {
        auto eventType = dep.getType().cast<EventType>();
        if (eventTypeGroups.count(eventType)) {
          eventTypeGroups[eventType].push_back(dep);
        } else {
          eventTypeGroups[eventType] = {dep};
        }
      }

      for (auto typeEvents : eventTypeGroups) {
        auto &eventType = typeEvents.first;
        auto &events = typeEvents.second;
        auto groupOp =
            builder.createOrFold<GroupEvents>(op->getLoc(), eventType, events);
        builder.createOrFold<Wait>(op->getLoc(), groupOp);
      }

      for (auto dep : dependees) {
        memoryEventMap.erase(dep);
      }
    }
  };

  funcOp.walk(gatherEventDependencies);
}

} // namespace

class ExecEnvCoalescing : public ExecEnvCoalescingBase<ExecEnvCoalescing> {
public:
  void runOnFunction() override {
    auto funcOp = getFunction();

    // Currently only single block is correctly optimized as optimizations rely
    // on knowing the order of execution
    // TODO Loosen this restriction
    if (funcOp.getBlocks().size() != 1)
      return;

    reuseExecEnv(funcOp);
    reuseMemory(funcOp);
    if (mlir::failed(eraseEventDependencies(funcOp))) {
      // Can't erase event dependencies, so optimizations stop here
      // TODO Extend following passes to work gracefully with existing
      // dependencies
      return;
    }
    removeRedundantRW(funcOp);
    recalculateEventDependencies(funcOp);
  }
};

std::unique_ptr<mlir::Pass> createExecEnvCoalescingPass() {
  return std::make_unique<ExecEnvCoalescing>();
}

} // namespace pmlc::dialect::comp
