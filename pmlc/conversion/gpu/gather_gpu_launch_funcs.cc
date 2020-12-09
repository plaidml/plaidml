// Copyright 2020 Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "pmlc/conversion/gpu/pass_detail.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::conversion::gpu {
namespace gpu = mlir::gpu;

struct Move {
  mlir::Operation *insertionOp;
  std::vector<mlir::Operation *> ops;
};

class GatherGpuLaunchFuncsPass
    : public GatherGpuLaunchFuncsPassBase<GatherGpuLaunchFuncsPass> {
public:
  void runOnOperation() override;

private:
  mlir::Operation *
  gatherAllocOpsStartingFromOp(mlir::Operation *op,
                               std::vector<mlir::Operation *> &ops);
  std::vector<std::shared_ptr<Move>>
  gatherAllocOpsBetweenConsecutiveLaunchOps(mlir::Block &block);
};

mlir::Operation *GatherGpuLaunchFuncsPass::gatherAllocOpsStartingFromOp(
    mlir::Operation *op, std::vector<mlir::Operation *> &ops) {
  while (op != nullptr) {
    if (mlir::isa<mlir::AllocOp>(op)) {
      ops.push_back(op);
    }
    if (!mlir::isa<mlir::AllocOp>(op) && !mlir::isa<gpu::LaunchFuncOp>(op)) {
      return op;
    }
    op = op->getNextNode();
  }
  return op;
}


std::vector<std::shared_ptr<Move>>
GatherGpuLaunchFuncsPass::gatherAllocOpsBetweenConsecutiveLaunchOps(
    mlir::Block &block) {
  std::vector<std::shared_ptr<Move>> moves;
  mlir::Operation *op = &(*block.begin());
  while (op != nullptr) {
    if (mlir::isa<gpu::LaunchFuncOp>(op)) {
      auto move = std::make_shared<Move>();
      move->insertionOp = op;
      op = gatherAllocOpsStartingFromOp(op, move->ops);
      moves.push_back(move);
    }
    if (op != nullptr) {
      op = op->getNextNode();
    }
  }
  return moves;
}

void GatherGpuLaunchFuncsPass::runOnOperation() {
  auto module = getOperation();
  module.walk([&](mlir::FuncOp funcOp) {
    if (funcOp.isExternal()) {
      return;
    }
    auto &blocks = funcOp.getBlocks();
    for (auto &block : blocks) {
      auto moves = gatherAllocOpsBetweenConsecutiveLaunchOps(block);
      for (auto move : moves) {
        for (auto &op : move->ops) {
          op->moveBefore(move->insertionOp);
        }
      }
    }
  });
}

std::unique_ptr<mlir::Pass> createGatherGpuLaunchFuncsPass() {
  return std::make_unique<GatherGpuLaunchFuncsPass>();
}
} // namespace pmlc::conversion::gpu
