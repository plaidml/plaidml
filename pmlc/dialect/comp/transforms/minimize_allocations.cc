// Copyright 2020 Intel Corporation

#include <unordered_map>

#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

#include "pmlc/dialect/comp/analysis/mem_sync_tracker.h"
#include "pmlc/dialect/comp/ir/dialect.h"
#include "pmlc/dialect/comp/transforms/pass_detail.h"
#include "pmlc/dialect/comp/transforms/passes.h"

namespace pmlc::dialect::comp {

namespace {

template <typename T>
class LLVMHash {
public:
  llvm::hash_code operator()(const T &val) const {
    return llvm::hash_value(val);
  }
};

/// Class tracking deallocated memory and finding reuse possibilities.
class DeallocatedMemoryTracker {
public:
  using EnvMemTypePair = std::pair<mlir::Value, mlir::MemRefType>;
  using DeallocMap = std::unordered_multimap<EnvMemTypePair, Dealloc,
                                             LLVMHash<EnvMemTypePair>>;
  using Iterator = DeallocMap::iterator;

  /// Marks memory as deallocated.
  void deallocMemory(Dealloc op) {
    EnvMemTypePair key = getDeallocKey(op);
    deallocated.emplace(key, op);
  }
  /// Finds any already deallocated memory that can be reused.
  /// Returns llvm::None if no memory can be reused.
  mlir::Optional<Iterator> findReuse(Alloc op) {
    EnvMemTypePair key = getAllocKey(op);
    DeallocMap::iterator it = deallocated.find(key);
    if (it == deallocated.end())
      return llvm::None;
    return it;
  }
  /// Returns range of deallocated memory that can be reused for
  /// specified allocation.
  std::pair<Iterator, Iterator> findAllReuses(Alloc op) {
    EnvMemTypePair key = getAllocKey(op);
    return deallocated.equal_range(key);
  }
  /// Removes memory pointed to by iterator from deallocated pool.
  void reuseMemory(Iterator it) { deallocated.erase(it); }

private:
  // Constructs deallocated map key from Alloc `op`.
  EnvMemTypePair getAllocKey(Alloc op) {
    mlir::Value execEnv = op.execEnv();
    auto memoryType = op.getType().cast<mlir::MemRefType>();
    return std::make_pair(execEnv, memoryType);
  }
  // Constructs deallocated map key from Dealloc `op`.
  EnvMemTypePair getDeallocKey(Dealloc op) {
    mlir::Value execEnv = op.execEnv();
    mlir::Value memory = op.deviceMem();
    auto memoryType = memory.getType().cast<mlir::MemRefType>();
    return std::make_pair(execEnv, memoryType);
  }

  DeallocMap deallocated;
};

// Replaces Alloc `op` with already allocated `memory`.
// Inserts copying of host memory if needed.
void replaceAlloc(Alloc op, mlir::Value memory, bool needsCopy = true) {
  mlir::Value hostMem = op.hostMem();
  if (hostMem && needsCopy) {
    // Write host content to reused memory.
    mlir::OpBuilder builder(op.getOperation());
    mlir::Value execEnv = op.execEnv();
    ExecEnvType execEnvType = execEnv.getType().cast<ExecEnvType>();
    EventType eventType = execEnvType.getEventType();
    mlir::Value event = builder.create<ScheduleWrite>(
        op.getLoc(), eventType, hostMem, memory, execEnv, mlir::ValueRange{});
    // Insert wait and let other passes clean it up.
    builder.create<Wait>(op.getLoc(), event);
  }
  op.replaceAllUsesWith(memory);
  op.erase();
}

/// Performs linear scan looking for memory that doesn't need
/// extra copy to be reused - new content and content at time
/// of deallocation is the same.
void inSyncReuse(mlir::Block &block) {
  DeallocatedMemoryTracker deallocTracker;
  MemorySynchronizationTracker syncTracker;
  for (mlir::Operation &operation : llvm::make_early_inc_range(block)) {
    llvm::TypeSwitch<mlir::Operation *>(&operation)
        .Case<Dealloc>([&](Dealloc op) {
          // No sync tracker, as we need to preserve information about
          // which memories are synchronized even after they are deallocated.
          deallocTracker.deallocMemory(op);
        })
        .Case<Alloc>([&](Alloc op) {
          if (!op.hostMem()) {
            syncTracker.handleAllocOp(op);
            return;
          }
          auto deallocRange = deallocTracker.findAllReuses(op);
          auto deallocIt = deallocRange.first;
          while (deallocIt != deallocRange.second) {
            Dealloc deallocOp = deallocIt->second;
            if (!syncTracker.areInSync(deallocOp.deviceMem(), op.hostMem())) {
              deallocIt++;
              continue;
            }
            // We found memory that is already in sync, reuse it.
            // No need to do extra copy, memory already has correct contents.
            replaceAlloc(op, deallocOp.deviceMem(), /*needsCopy=*/false);
            deallocOp.erase();
            deallocTracker.reuseMemory(deallocIt);
            return;
          }
          // If no reuse is possible track synchronization status of allocation.
          syncTracker.handleAllocOp(op);
        })
        .Default([&](mlir::Operation *op) { syncTracker.handleOperation(op); });
  }
}

/// Performs very greedy linear scan for memory reuse opportunities.
/// Does not check any interference for Out-of-Order execution.
void greedyInPlaceReuse(mlir::Block &block) {
  DeallocatedMemoryTracker deallocTracker;
  for (mlir::Operation &operation : llvm::make_early_inc_range(block)) {
    if (auto deallocOp = mlir::dyn_cast<Dealloc>(operation)) {
      deallocTracker.deallocMemory(deallocOp);
    }
    if (auto allocOp = mlir::dyn_cast<Alloc>(operation)) {
      auto optReuseIt = deallocTracker.findReuse(allocOp);
      if (!optReuseIt.hasValue())
        continue;
      DeallocatedMemoryTracker::Iterator reuseIt = optReuseIt.getValue();
      Dealloc deallocOp = reuseIt->second;
      replaceAlloc(allocOp, deallocOp.deviceMem());
      deallocOp.erase();
      deallocTracker.reuseMemory(reuseIt);
    }
  }
}

class MinimizeAllocationsPass final
    : public MinimizeAllocationsBase<MinimizeAllocationsPass> {
public:
  void runOnFunction() {
    mlir::FuncOp func = getFunction();
    for (mlir::Block &block : func) {
      // First try to reuse memory already in sync,
      // then find all other reuse possibilites.
      inSyncReuse(block);
      greedyInPlaceReuse(block);
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createMinimizeAllocationsPass() {
  return std::make_unique<MinimizeAllocationsPass>();
}

} // namespace pmlc::dialect::comp
