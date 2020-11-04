// Copyright 2020 Intel Corporation

#include "pmlc/dialect/comp/analysis/mem_sync_tracker.h"

#include "llvm/ADT/TypeSwitch.h"

namespace pmlc::dialect::comp {

bool MemorySynchronizationTracker::areInSync(mlir::Value a, mlir::Value b) {
  auto inSyncIt = inSyncMap.find(a);
  if (inSyncIt == inSyncMap.end())
    return false;
  return inSyncIt->second.count(b);
}

bool MemorySynchronizationTracker::isTracked(mlir::Value mem) {
  return inSyncMap.count(mem) > 0;
}

bool MemorySynchronizationTracker::handleOperation(mlir::Operation *operation) {
  return llvm::TypeSwitch<mlir::Operation *, bool>(operation)
      .Case<MemoryTransferOpInterface>(
          [&](MemoryTransferOpInterface op) { return handleTransferOp(op); })
      .Case<Alloc>([&](Alloc op) { return handleAllocOp(op); })
      .Default([&](mlir::Operation *op) { return handleGeneralOp(op); });
}

bool MemorySynchronizationTracker::handleTransferOp(
    MemoryTransferOpInterface op) {
  mlir::Value src = op.getSource();
  mlir::Value dst = op.getDestination();
  return syncMemory(src, dst);
}

bool MemorySynchronizationTracker::handleAllocOp(Alloc op) {
  if (mlir::Value hostMem = op.hostMem())
    return syncMemory(hostMem, op.getResult());
  return false;
}

bool MemorySynchronizationTracker::handleGeneralOp(mlir::Operation *op) {
  bool changed = false;
  // Find memory used and mark as desynchronized.
  for (mlir::Value operand : op->getOperands())
    if (operand.getType().isa<mlir::MemRefType>())
      changed |= desyncMemory(operand);
  return changed;
}

bool MemorySynchronizationTracker::syncMemory(mlir::Value src,
                                              mlir::Value dst) {
  if (areInSync(src, dst))
    return false;
  desyncMemory(dst);
  for (mlir::Value srcInSync : inSyncMap[src]) {
    inSyncMap[dst].insert(srcInSync);
    inSyncMap[srcInSync].insert(dst);
  }
  inSyncMap[dst].insert(src);
  inSyncMap[src].insert(dst);
  return true;
}

bool MemorySynchronizationTracker::desyncMemory(mlir::Value mem) {
  auto memSyncIt = inSyncMap.find(mem);
  if (memSyncIt == inSyncMap.end())
    return false;
  if (memSyncIt->second.empty())
    return false;
  for (mlir::Value inSync : memSyncIt->second)
    inSyncMap[inSync].erase(mem);
  memSyncIt->second.clear();
  return true;
}

} // namespace pmlc::dialect::comp
