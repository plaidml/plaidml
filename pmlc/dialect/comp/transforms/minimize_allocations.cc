// Copyright 2020 Intel Corporation

#include <unordered_map>

#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

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

/// Performs very greedy linear scan for memory reuse opportunities.
/// Does not check any interference for Out-of-Order execution.
void greedyInPlaceReuse(mlir::Block &block) {
  auto builder = mlir::OpBuilder::atBlockBegin(&block);
  // Replaces Alloc `op` with already allocated `memory`.
  // Inserts copying of host memory if needed.
  auto replaceAlloc = [&](Alloc op, mlir::Value memory) {
    if (mlir::Value hostMem = op.hostMem()) {
      builder.setInsertionPoint(op);
      mlir::Value execEnv = op.execEnv();
      ExecEnvType execEnvType = execEnv.getType().cast<ExecEnvType>();
      EventType eventType = execEnvType.getEventType();
      mlir::Value event = builder.create<ScheduleWrite>(
          op.getLoc(), eventType, hostMem, memory, execEnv, mlir::ValueRange{});
      builder.create<Wait>(op.getLoc(), event);
    }
    op.replaceAllUsesWith(memory);
    op.erase();
  };
  // Map used to store deallocated memory. After processing Dealloc op it is
  // assumed that further operations can safely reuse it.
  using EnvMemTypePair = std::pair<mlir::Value, mlir::MemRefType>;
  std::unordered_multimap<EnvMemTypePair, Dealloc, LLVMHash<EnvMemTypePair>>
      deallocated;
  // Constructs deallocated map key from Dealloc `op`.
  auto getDeallocKey = [](Dealloc op) {
    mlir::Value execEnv = op.execEnv();
    mlir::Value memory = op.deviceMem();
    auto memoryType = memory.getType().cast<mlir::MemRefType>();
    return std::make_pair(execEnv, memoryType);
  };
  // Constructs deallocated map key from Alloc `op`.
  auto getAllocKey = [](Alloc op) {
    mlir::Value execEnv = op.execEnv();
    auto memoryType = op.getType().cast<mlir::MemRefType>();
    return std::make_pair(execEnv, memoryType);
  };

  for (mlir::Operation &operation : llvm::make_early_inc_range(block)) {
    if (auto deallocOp = mlir::dyn_cast<Dealloc>(operation)) {
      EnvMemTypePair key = getDeallocKey(deallocOp);
      deallocated.emplace(key, deallocOp);
    }
    if (auto allocOp = mlir::dyn_cast<Alloc>(operation)) {
      EnvMemTypePair key = getAllocKey(allocOp);
      auto deallocIt = deallocated.find(key);
      if (deallocIt != deallocated.end()) {
        Dealloc deallocOp = deallocIt->second;
        replaceAlloc(allocOp, deallocOp.deviceMem());
        deallocOp.erase();
        deallocated.erase(deallocIt);
      }
    }
  }
}

class MinimizeAllocationsPass final
    : public MinimizeAllocationsBase<MinimizeAllocationsPass> {
public:
  void runOnFunction() {
    mlir::FuncOp func = getFunction();
    for (mlir::Block &block : func) {
      greedyInPlaceReuse(block);
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createMinimizeAllocationsPass() {
  return std::make_unique<MinimizeAllocationsPass>();
}

} // namespace pmlc::dialect::comp
