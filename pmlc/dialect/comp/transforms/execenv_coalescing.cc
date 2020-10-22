// Copyright 2020 Intel Corporation

#include <utility>

#include "mlir/IR/Block.h"
#include "mlir/IR/Function.h"

#include "pmlc/dialect/comp/ir/dialect.h"
#include "pmlc/dialect/comp/transforms/pass_detail.h"
#include "pmlc/dialect/comp/transforms/passes.h"

namespace pmlc::dialect::comp {

namespace {
/// Finds earliest creation of exec environment with each type and device
/// and latest destruction. Reuse environment from first creation and remove
/// all destructions except last.
/// Blocks that have execution environment going in or leaving are
/// not optimized.
/// Aliases for device are not analyzed.
void reuseExecEnvInBlock(mlir::Block &block) {
  // TODO: Relax requirement of no incoming or outgoing execenvs.
  // Check that all creations are local to this block.
  for (mlir::Type argType : block.getArgumentTypes()) {
    if (argType.isa<ExecEnvType>())
      return;
  }
  // Check that all destructions are local to this block.
  for (mlir::Block *successor : block.getSuccessors()) {
    for (mlir::Type succArgType : successor->getArgumentTypes()) {
      if (succArgType.isa<ExecEnvType>())
        return;
    }
  }
  // Find earliest creation and lastest destruction of each type and device.
  using TypeDevicePair = std::pair<mlir::Type, mlir::Value>;
  llvm::DenseMap<TypeDevicePair, CreateExecEnv> earliestCreation;
  llvm::DenseMap<TypeDevicePair, DestroyExecEnv> latestDestruction;
  auto getCreateKey = [](CreateExecEnv op) -> TypeDevicePair {
    mlir::Type execEnvType = op.getType();
    mlir::Value device = op.getOperand();
    return std::make_pair(execEnvType, device);
  };
  auto getDestroyKey = [](DestroyExecEnv op) -> TypeDevicePair {
    mlir::Value execEnv = op.getOperand();
    mlir::Type execEnvType = execEnv.getType();
    auto createOp = execEnv.getDefiningOp<CreateExecEnv>();
    mlir::Value device = createOp.getOperand();
    return std::make_pair(execEnvType, device);
  };

  for (mlir::Operation &op : block) {
    if (auto createOp = mlir::dyn_cast<CreateExecEnv>(op)) {
      TypeDevicePair key = getCreateKey(createOp);
      if (earliestCreation.count(key) == 0)
        earliestCreation[key] = createOp;
    }
    if (auto destroyOp = mlir::dyn_cast<DestroyExecEnv>(op)) {
      TypeDevicePair key = getDestroyKey(destroyOp);
      latestDestruction[key] = destroyOp;
    }
  }
  // Replace all non-earliest creations and erase non-latest destructions.
  for (mlir::Operation &op : llvm::make_early_inc_range(block)) {
    if (auto createOp = mlir::dyn_cast<CreateExecEnv>(op)) {
      TypeDevicePair key = getCreateKey(createOp);
      if (earliestCreation[key] != createOp) {
        op.replaceAllUsesWith(earliestCreation[key]);
        op.erase();
      }
    } else if (auto destroyOp = mlir::dyn_cast<DestroyExecEnv>(op)) {
      TypeDevicePair key = getDestroyKey(destroyOp);
      if (latestDestruction[key] != destroyOp)
        op.erase();
    }
  }
}

class ExecEnvCoalescingPass final
    : public ExecEnvCoalescingBase<ExecEnvCoalescingPass> {
public:
  void runOnFunction() override {
    mlir::Region &region = getFunction().getRegion();
    for (mlir::Block &block : region.getBlocks())
      reuseExecEnvInBlock(block);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createExecEnvCoalescingPass() {
  return std::make_unique<ExecEnvCoalescingPass>();
}

} // namespace pmlc::dialect::comp
