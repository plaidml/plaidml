// Copyright 2020 Intel Corporation

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

#include "pmlc/dialect/comp/ir/dialect.h"
#include "pmlc/dialect/comp/transforms/pass_detail.h"
#include "pmlc/dialect/comp/transforms/passes.h"

namespace pmlc::dialect::comp {

namespace {

struct EventDependencyRecalculator {
  explicit EventDependencyRecalculator(bool safeDealloc)
      : safeDealloc(safeDealloc) {}

  using SmallValueSet = llvm::SmallDenseSet<mlir::Value, 4>;

  /// Performs dependency recalulcation for specified block.
  void recalculate(mlir::Block &block);

  /// Gather memref values present as operands of operation `op` or any
  /// other nested operation.
  SmallValueSet getMemoryUsed(mlir::Operation *op);
  /// Inserts operation before `op` that waits for `events` to finish.
  void createWait(mlir::Operation *op, const SmallValueSet &events);

  /// Methods to handle different operations.
  void handleDeallocOp(Dealloc op);
  void handleDestroyExecEnvOp(DestroyExecEnv op);
  void handleScheduleOp(ScheduleOpInterface op);
  void handleWaitOp(Wait op);
  void handleGeneralOp(mlir::Operation *op);

  /// Insert explicit wait before dealloc.
  bool safeDealloc;
  /// Mapping between memory and event representing last operation on
  /// that memory.
  mlir::DenseMap<mlir::Value, mlir::Value> memoryEventMap;
  /// Set of all events that are tracked by recalculator.
  mlir::DenseSet<mlir::Value> trackedEvents;
  /// Set of events that were not used as dependency for any other event
  /// or explicitly waited for.
  mlir::DenseSet<mlir::Value> eventFront;
};

void EventDependencyRecalculator::recalculate(mlir::Block &block) {
  memoryEventMap.clear();
  trackedEvents.clear();
  eventFront.clear();
  for (mlir::Operation &operation : llvm::make_early_inc_range(block)) {
    llvm::TypeSwitch<mlir::Operation *, void>(&operation)
        .Case<ScheduleOpInterface>(
            [&](ScheduleOpInterface op) { handleScheduleOp(op); })
        .Case<Wait>([&](Wait op) { handleWaitOp(op); })
        .Case<Dealloc>([&](Dealloc op) { handleDeallocOp(op); })
        .Case<DestroyExecEnv>(
            [&](DestroyExecEnv op) { handleDestroyExecEnvOp(op); })
        .Default([&](mlir::Operation *op) { handleGeneralOp(op); });
  }
  if (!eventFront.empty()) {
    // Wait for any leftover events before terminator.
    SmallValueSet waitFor(eventFront.begin(), eventFront.end());
    createWait(block.getTerminator(), waitFor);
  }
}

void EventDependencyRecalculator::handleWaitOp(Wait op) {
  // Preserve events that are not tracked by recalculator, ie. come from outside
  // of block.
  mlir::SmallVector<mlir::Value, 1> untrackedEvents;
  for (mlir::Value event : op.events()) {
    if (trackedEvents.count(event) == 0)
      untrackedEvents.push_back(event);
  }
  if (untrackedEvents.empty())
    op.erase();
  else
    op.eventsMutable().assign(untrackedEvents);
}

EventDependencyRecalculator::SmallValueSet
EventDependencyRecalculator::getMemoryUsed(mlir::Operation *operation) {
  SmallValueSet memoryUsed;
  // Gather from direct operands.
  for (mlir::Value operand : operation->getOperands()) {
    if (operand.getType().isa<mlir::MemRefType>())
      memoryUsed.insert(operand);
  }
  // Append operands of nested operations.
  for (mlir::Region &region : operation->getRegions()) {
    region.walk([&](mlir::Operation *op) {
      for (mlir::Value operand : op->getOperands()) {
        // Discard memory that is created inside this region.
        if (operand.getType().isa<mlir::MemRefType>() &&
            !region.isProperAncestor(operand.getParentRegion()))
          memoryUsed.insert(operand);
      }
    });
  }
  return memoryUsed;
}

void EventDependencyRecalculator::createWait(mlir::Operation *op,
                                             const SmallValueSet &events) {
  if (events.empty())
    return;
  mlir::OpBuilder builder(op);
  mlir::SmallVector<mlir::Value, 4> eventsVec(events.begin(), events.end());
  builder.create<Wait>(op->getLoc(), eventsVec);
}

void EventDependencyRecalculator::handleDeallocOp(Dealloc op) {
  // With no safeDealloc dependencies can be ignored, otherwise
  // handle same as normal operation.
  if (safeDealloc)
    handleGeneralOp(op.getOperation());
}

void EventDependencyRecalculator::handleDestroyExecEnvOp(DestroyExecEnv op) {
  // TODO: This should be done in a smarter way to not wait for
  //       events from other execution environments.
  llvm::SmallDenseSet<mlir::Value, 4> waitFor;
  for (mlir::Value event : eventFront)
    waitFor.insert(event);

  createWait(op.getOperation(), waitFor);
  memoryEventMap.clear();
  eventFront.clear();
  trackedEvents.clear();
}

void EventDependencyRecalculator::handleScheduleOp(ScheduleOpInterface op) {
  SmallValueSet memoryUsed = getMemoryUsed(op.getOperation());
  // Map memory into corresponding events.
  SmallValueSet waitFor;
  for (mlir::Value memory : memoryUsed) {
    auto it = memoryEventMap.find(memory);
    if (it != memoryEventMap.end())
      waitFor.insert(it->second);
  }
  // Merge existing dependencies with recalculated ones.
  for (mlir::Value event : op.getDependencies()) {
    if (trackedEvents.count(event) > 0)
      continue;
    waitFor.insert(event);
  }
  // Update list of operations dependencies.
  mlir::SmallVector<mlir::Value, 4> waitForVec(waitFor.begin(), waitFor.end());
  op.getDependencyMutable().assign(waitForVec);
  // Update tracked events, memory->event map and event front.
  mlir::Value resultingEvent = op.getResultingEvent();
  trackedEvents.insert(resultingEvent);
  for (mlir::Value memory : memoryUsed)
    memoryEventMap[memory] = resultingEvent;
  for (mlir::Value event : waitFor)
    eventFront.erase(event);
  eventFront.insert(resultingEvent);
}

void EventDependencyRecalculator::handleGeneralOp(mlir::Operation *op) {
  SmallValueSet memoryUsed = getMemoryUsed(op);
  if (memoryUsed.empty())
    return;
  // Map memory into corresponding events.
  SmallValueSet waitFor;
  for (mlir::Value memory : memoryUsed) {
    auto it = memoryEventMap.find(memory);
    if (it != memoryEventMap.end())
      waitFor.insert(it->second);
  }
  // Insert wait before.
  createWait(op, waitFor);
  // Remove from memory->event map and event front as already waited for.
  for (mlir::Value memory : memoryUsed)
    memoryEventMap.erase(memory);
  for (mlir::Value event : waitFor)
    eventFront.erase(event);
}

class RecalculateEventDepsPass final
    : public RecalculateEventDepsBase<RecalculateEventDepsPass> {
public:
  RecalculateEventDepsPass() = default;
  explicit RecalculateEventDepsPass(bool safeDealloc) {
    this->safeDealloc = safeDealloc;
  }

  void runOnFunction();
};

void RecalculateEventDepsPass::runOnFunction() {
  mlir::FuncOp func = getFunction();
  EventDependencyRecalculator recalc(safeDealloc.getValue());
  for (mlir::Block &block : func)
    recalc.recalculate(block);
}

} // namespace

std::unique_ptr<mlir::Pass> createRecalculateEventDepsPass() {
  return std::make_unique<RecalculateEventDepsPass>();
}

std::unique_ptr<mlir::Pass> createRecalculateEventDepsPass(bool safeDealloc) {
  return std::make_unique<RecalculateEventDepsPass>(safeDealloc);
}

} // namespace pmlc::dialect::comp
